# ATHENA-DS Dashboard

Este é um painel interativo para descoberta de fármacos usando dados do ChEMBL, descritores moleculares e inteligência artificial com SHAP.

## Como usar

1. Suba este repositório no GitHub (crie um novo repo e envie os arquivos).
2. Crie uma conta no [https://streamlit.io](https://streamlit.io)
3. Vá em "Deploy an app", conecte seu GitHub e selecione este repositório.
4. Como script inicial, use `athena_dashboard.py`.

Seu dashboard estará no ar em minutos!

---

**Requisitos:** Python 3.8+ com as bibliotecas em `requirements.txt`

## Script para verificar LLMs disponíveis no GitHub Models

Adicionei o script `github_models_quota_check.py` para listar os modelos do catálogo do GitHub Models e, opcionalmente, testar quais ainda respondem com a cota atual do seu token.

### Uso

```bash
export GITHUB_TOKEN=seu_token_com_models_read
python github_models_quota_check.py --probe
```

### Filtros úteis

```bash
# Ver só modelos da OpenAI
python github_models_quota_check.py --probe --publisher OpenAI

# Testar só um modelo específico
python github_models_quota_check.py --probe --model openai/gpt-4.1

# Atribuir o consumo a uma organização
python github_models_quota_check.py --probe --org SUA_ORG

# Saída JSON
python github_models_quota_check.py --probe --json
```

### O que o status significa

- `available`: o modelo respondeu a uma inferência mínima com o token atual.
- `quota_exceeded` / `quota_or_rate_limited`: o modelo existe, mas sua conta/token atingiu a cota ou o limite de taxa.
- `auth_error` / `forbidden`: problema de permissão, escopo ou acesso.
- `unsupported_for_probe`: o modelo apareceu no catálogo, mas não aceitou esse probe simples.
- `catalog_only`: apenas listou o catálogo, sem testar inferência.

