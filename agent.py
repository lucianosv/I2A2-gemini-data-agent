# agent.py

import io
import os
import contextlib
import traceback
import pandas as pd
import streamlit as st
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def initialize_gemini_api(api_key: str):
    """Inicializa a API do Gemini com a chave fornecida."""
    genai.configure(api_key=api_key)

# ---------------- Classificação de intenção ----------------
def classify_intent(user_prompt: str) -> str:
    """
    Usa IA para classificar intenção em 'analysis' ou 'chat'.
    Retorna exatamente 'analysis' ou 'chat'; em erro, retorna 'chat' como fallback seguro.
    """
    try:
#        model = genai.GenerativeModel("gemini-2.5-pro")
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
Você é um classificador de intenções. Analise a pergunta e responda APENAS com:
- analysis: se o usuário pedir para analisar, plotar, descrever, correlacionar ou consultar dados.
- chat: se for saudação, conversa geral, metaperguntas, agradecimentos, etc.

Pergunta: {user_prompt}

Responda com UMA palavra: analysis ou chat.
        """
        resp = model.generate_content(prompt)
        text = (resp.text or "").strip().lower()
        
        if "analysis" in text and "chat" not in text:
            return "analysis"
        if "chat" in text and "analysis" not in text:
            return "chat"
        
        # fallback simples por palavras-chave
        return (
            "analysis"
            if any(
                k in user_prompt.lower()
                for k in ["analis", "plot", "gráfico", "grafico", "describe", "correla", "hist", "box", "scatter"]
            )
            else "chat"
        )
    except Exception:
        return "chat"

# ---------------- Resposta de chat ----------------
def get_chat_response(user_prompt: str) -> str:
    try:
        #model = genai.GenerativeModel("gemini-2.5-pro")
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        sys = """Você é o Gemini Data Agent. Responda de forma clara e amigável.
Se a pergunta não exigir análise de dados, seja breve."""
        resp = model.generate_content([sys, user_prompt])
        return (resp.text or "").strip()
    except Exception as e:
        return f"Não foi possível gerar uma resposta de chat: {e}"

# ---------------- Geração de código de análise ----------------
def get_analysis_code(user_prompt: str, sample_markdown: str) -> str:
    """
    Retorna APENAS código Python que usa o DataFrame 'df' já existente.
    Deve imprimir alguma saída textual e, quando possível, um 'INSIGHT: ...' ao final.
    """
    try:
        #model = genai.GenerativeModel("gemini-2.5-pro")
        model = genai.GenerativeModel("gemini-2.0-flash-lite")
        prompt = f"""
Você irá gerar APENAS código Python (sem explicações ou cercas ```

Regras:
- O DataFrame já existe como df (pandas). NÃO leia arquivos.
- Se converter tipos, trate NaN previamente (ex.: fillna, dropna) antes de astype(int).
- Para Matplotlib/Seaborn: chame st.pyplot(plt.gcf()) após o plot.
- Para Plotly: use st.plotly_chart(fig, use_container_width=True).
- Evite chained assignment; use df.loc[...].
- Mostre prints com resultados e métricas relevantes (print()).
- Ao final, imprima uma linha começando com 'INSIGHT:' resumindo a principal conclusão (máx. 140 caracteres).
- Você pode criar novos DataFrames auxiliares se necessário.

Contexto (amostra df em Markdown):
{sample_markdown}

Tarefa do usuário:
{user_prompt}
        """
        resp = model.generate_content(prompt)
        code = (resp.text or "").strip()
        
        # Remove cercas acidentais
        if code.startswith("```"):
            code = code.strip("`")  # remove crases extras
        if code.lstrip().lower().startswith("python"):
            code = code.split("\n", 1)[-1]
            
        return code
    except Exception:
        # fallback mínimo para não quebrar o app
        return (
            "print('Falha ao gerar código. Mostrando preview do df:')\n"
            "print(df.head())\n"
            "print('INSIGHT: Não foi possível gerar a análise solicitada; verifique sua conexão ou reformule o pedido.')\n"
        )

# ---------------- Execução do código ----------------
def execute_code(code: str, df: pd.DataFrame):
    """
    Executa o código gerado em um namespace controlado com acesso a:
    df, st, pd, plt, sns, px.
    Retorna (stdout, error_text). Gráficos são exibidos via Streamlit no próprio código.
    """
    # CORREÇÃO APLICADA AQUI:
    safe_globals = {
        "__builtins__": {
            "__import__": __import__,  # ADICIONADO - essencial para importações
            "abs": abs,
            "min": min,
            "max": max,
            "sum": sum,
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "sorted": sorted,
            "round": round,
            "print": print,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "bool": bool,
            "type": type,
            "isinstance": isinstance,
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "map": map,
            "filter": filter,
            "any": any,
            "all": all,
        },
        "pd": pd,
        "st": st,
        "plt": plt,
        "sns": sns,
        "px": px,
    }
    
    safe_locals = {"df": df}
    stdout_buffer = io.StringIO()
    error_text = ""
    
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code, safe_globals, safe_locals)
    except Exception:
        error_text = traceback.format_exc()
    
    return stdout_buffer.getvalue(), error_text
