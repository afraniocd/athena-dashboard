#!/usr/bin/env python3
"""Lista modelos do GitHub Models e testa quais ainda estão acessíveis na cota atual.

Uso rápido:
    export GITHUB_TOKEN=ghp_xxx
    python github_models_quota_check.py --probe

O script:
1. Busca o catálogo em https://models.github.ai/catalog/models
2. Opcionalmente envia uma inferência mínima para cada modelo textual
3. Marca cada modelo como disponível, sem cota, sem permissão ou incompatível
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from typing import Any

import requests

API_VERSION = "2026-03-10"
CATALOG_URL = "https://models.github.ai/catalog/models"
INFERENCE_URL = "https://models.github.ai/inference/chat/completions"
ORG_INFERENCE_URL = "https://models.github.ai/orgs/{org}/inference/chat/completions"
DEFAULT_TIMEOUT = 30


@dataclass
class ProbeResult:
    model_id: str
    status: str
    http_status: int | None
    detail: str
    rate_limit_tier: str | None
    capabilities: list[str]
    input_modalities: list[str]
    output_modalities: list[str]


class GitHubModelsError(RuntimeError):
    """Erro controlado para chamadas da API do GitHub Models."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Lista os modelos do GitHub Models e, opcionalmente, testa quais ainda "
            "estão acessíveis com o token/cota atual."
        )
    )
    parser.add_argument(
        "--token",
        default=os.getenv("GITHUB_TOKEN"),
        help="GitHub token com permissão models:read. Padrão: variável GITHUB_TOKEN.",
    )
    parser.add_argument(
        "--org",
        help="Atribui as inferências a uma organização específica (usa /orgs/{org}/...).",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Envia uma inferência mínima para verificar se o modelo ainda está utilizável.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limita a quantidade de modelos processados. 0 = sem limite.",
    )
    parser.add_argument(
        "--publisher",
        action="append",
        default=[],
        help="Filtra por publisher. Pode ser repetido. Ex: --publisher OpenAI",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Filtra por ID exato do modelo. Pode ser repetido.",
    )
    parser.add_argument(
        "--include-non-text",
        action="store_true",
        help="Inclui modelos sem saída textual na listagem. O probe continua só para modelos textuais.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8,
        help="Quantidade máxima de tokens na inferência de teste. Padrão: 8.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Pausa em segundos entre probes, para evitar rate limit. Padrão: 0.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime o resultado em JSON.",
    )
    return parser.parse_args()


def build_headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": API_VERSION,
        "Content-Type": "application/json",
        "User-Agent": "athena-dashboard-github-models-checker",
    }


def extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason

    if isinstance(payload, dict):
        for key in ("error", "message", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        errors = payload.get("errors")
        if isinstance(errors, list) and errors:
            first = errors[0]
            if isinstance(first, dict):
                return json.dumps(first, ensure_ascii=False)
            return str(first)

    return json.dumps(payload, ensure_ascii=False)


def fetch_catalog(headers: dict[str, str], timeout: int = DEFAULT_TIMEOUT) -> list[dict[str, Any]]:
    response = requests.get(CATALOG_URL, headers=headers, timeout=timeout)
    if response.status_code != 200:
        raise GitHubModelsError(
            f"Falha ao buscar catálogo ({response.status_code}): {extract_error_message(response)}"
        )

    payload = response.json()
    if not isinstance(payload, list):
        raise GitHubModelsError("Resposta inesperada do catálogo: era esperado um array de modelos.")
    return payload


def filter_models(models: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    publishers = {publisher.casefold() for publisher in args.publisher}
    selected_models = set(args.model)

    filtered = []
    for model in models:
        publisher = str(model.get("publisher", ""))
        model_id = str(model.get("id", ""))
        output_modalities = model.get("supported_output_modalities") or []

        if publishers and publisher.casefold() not in publishers:
            continue
        if selected_models and model_id not in selected_models:
            continue
        if not args.include_non_text and "text" not in output_modalities:
            continue

        filtered.append(model)

    if args.limit > 0:
        return filtered[: args.limit]
    return filtered


def classify_probe_response(response: requests.Response) -> tuple[str, str]:
    detail = extract_error_message(response)
    lowered = detail.casefold()

    if response.status_code == 200:
        return "available", "Inferência executada com sucesso."
    if response.status_code in {401}:
        return "auth_error", detail
    if response.status_code in {402}:
        return "quota_exceeded", detail
    if response.status_code in {403, 429}:
        if any(token in lowered for token in ("quota", "rate limit", "limit exceeded", "too many requests")):
            return "quota_or_rate_limited", detail
        return "forbidden", detail
    if response.status_code == 422:
        return "unsupported_for_probe", detail
    return "error", detail


def run_probe(
    model: dict[str, Any],
    headers: dict[str, str],
    args: argparse.Namespace,
    timeout: int = DEFAULT_TIMEOUT,
) -> ProbeResult:
    model_id = str(model.get("id", ""))
    capabilities = list(model.get("capabilities") or [])
    input_modalities = list(model.get("supported_input_modalities") or [])
    output_modalities = list(model.get("supported_output_modalities") or [])
    rate_limit_tier = model.get("rate_limit_tier")

    if "text" not in output_modalities:
        return ProbeResult(
            model_id=model_id,
            status="not_probed",
            http_status=None,
            detail="Modelo sem saída textual; probe pulado.",
            rate_limit_tier=rate_limit_tier,
            capabilities=capabilities,
            input_modalities=input_modalities,
            output_modalities=output_modalities,
        )

    url = ORG_INFERENCE_URL.format(org=args.org) if args.org else INFERENCE_URL
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Responda apenas OK"}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    status, detail = classify_probe_response(response)

    return ProbeResult(
        model_id=model_id,
        status=status,
        http_status=response.status_code,
        detail=detail,
        rate_limit_tier=rate_limit_tier,
        capabilities=capabilities,
        input_modalities=input_modalities,
        output_modalities=output_modalities,
    )


def format_table(results: list[ProbeResult]) -> str:
    rows = [
        [
            result.model_id,
            result.status,
            str(result.http_status or "-"),
            result.rate_limit_tier or "-",
            ",".join(result.output_modalities) or "-",
            result.detail.replace("\n", " ")[:110],
        ]
        for result in results
    ]

    headers = ["model", "status", "http", "tier", "output", "detail"]
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))

    def render(row: list[str]) -> str:
        return " | ".join(cell.ljust(widths[idx]) for idx, cell in enumerate(row))

    separator = "-+-".join("-" * width for width in widths)
    lines = [render(headers), separator]
    lines.extend(render(row) for row in rows)
    return "\n".join(lines)


def to_probe_result(model: dict[str, Any]) -> ProbeResult:
    return ProbeResult(
        model_id=str(model.get("id", "")),
        status="catalog_only",
        http_status=None,
        detail="Modelo listado no catálogo; probe não executado.",
        rate_limit_tier=model.get("rate_limit_tier"),
        capabilities=list(model.get("capabilities") or []),
        input_modalities=list(model.get("supported_input_modalities") or []),
        output_modalities=list(model.get("supported_output_modalities") or []),
    )


def main() -> int:
    args = parse_args()
    if not args.token:
        print(
            "Erro: informe --token ou defina GITHUB_TOKEN com permissão models:read.",
            file=sys.stderr,
        )
        return 2

    headers = build_headers(args.token)

    try:
        models = fetch_catalog(headers)
    except GitHubModelsError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Erro de rede ao consultar o catálogo: {exc}", file=sys.stderr)
        return 1

    filtered_models = filter_models(models, args)
    if not filtered_models:
        print("Nenhum modelo encontrado com os filtros informados.")
        return 0

    results: list[ProbeResult] = []
    for model in filtered_models:
        try:
            result = run_probe(model, headers, args) if args.probe else to_probe_result(model)
        except requests.RequestException as exc:
            result = ProbeResult(
                model_id=str(model.get("id", "")),
                status="network_error",
                http_status=None,
                detail=str(exc),
                rate_limit_tier=model.get("rate_limit_tier"),
                capabilities=list(model.get("capabilities") or []),
                input_modalities=list(model.get("supported_input_modalities") or []),
                output_modalities=list(model.get("supported_output_modalities") or []),
            )
        results.append(result)
        if args.probe and args.sleep > 0:
            time.sleep(args.sleep)

    if args.json:
        print(
            json.dumps(
                [result.__dict__ for result in results],
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        print(format_table(results))

        available_count = sum(result.status == "available" for result in results)
        print()
        print(f"Total no catálogo/seleção: {len(results)}")
        if args.probe:
            print(f"Disponíveis com sucesso no probe: {available_count}")
            print("Observação: 'available' significa que a chamada mínima respondeu agora com esse token.")
            print(
                "Observação: 'quota_or_rate_limited' indica que o modelo existe, mas a conta/token bateu em limite ou cota."
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
