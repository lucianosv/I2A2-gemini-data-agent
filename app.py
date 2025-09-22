# app.py

import os
import io
import pandas as pd
import streamlit as st
from agent import (
    initialize_gemini_api,
    classify_intent,
    get_analysis_code,
    get_chat_response,
    execute_code,
)

# ---------------- Configuração da página ----------------
st.set_page_config(page_title="Agente de Análise de Dados", page_icon="📊", layout="wide")

# ---------------- Estado da sessão ----------------
st.session_state.setdefault("df", None)
st.session_state.setdefault("file_meta", None) # (name, size)
st.session_state.setdefault("sample_rendered", False) # controla render da amostra
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("insights", []) # memória de conclusões

# ---------------- Chave de API ----------------
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("A chave de API do Gemini não foi encontrada. Configure a variável de ambiente 'GEMINI_API_KEY'.")
    st.stop()
else:
    initialize_gemini_api(api_key)

# ---------------- UI ----------------
st.title("🎲 Gemini Data Agent")
st.markdown("Faça o upload de um arquivo CSV e comece a conversar com o agente de IA.")

# Upload
uploaded_file = st.file_uploader("Escolha um arquivo CSV", type=["csv"])

# Detecta mudança de arquivo com metadados estáveis
def get_meta(f):
    if f is None:
        return None
    try:
        size = f.size
    except Exception:
        try:
            size = len(f.getbuffer())
        except Exception:
            size = None
    return (f.name, size)

new_meta = get_meta(uploaded_file)

if uploaded_file is not None:
    if st.session_state.file_meta != new_meta:
        # Novo arquivo: carrega df, reseta amostra e dá boas-vindas
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.file_meta = new_meta
            st.session_state.sample_rendered = False
            st.session_state.chat_history = [
                {"role": "assistant", "content": "Arquivo recebido! Pronto para analisar. Faça uma pergunta ou peça um gráfico."}
            ]
            st.session_state.insights = [] # reinicia memória para novo conjunto
        except Exception as e:
            st.error(f"Falha ao ler CSV: {e}")
            st.stop()
else:
    st.info("Envie um CSV para começar.")
    st.stop()

# Layout principal
#left, right = st.columns([2, 1])

#with left:
st.subheader("Amostra do DataFrame")
sample_box = st.empty()

if st.session_state.df is not None and not st.session_state.sample_rendered:
    sample_box.dataframe(st.session_state.df.head(10)) # primeira renderização
    st.session_state.sample_rendered = True
elif st.session_state.sample_rendered:
    # Mantém a amostra visível entre reruns
    sample_box.dataframe(st.session_state.df.head(10))

#with right:
st.subheader("Memória de conclusões")
if st.session_state.insights:
    for i, ins in enumerate(st.session_state.insights, 1):
        st.markdown(f"- {ins}")
else:
    st.caption("Sem conclusões registradas ainda. Peça análises e gráficos.")

# Histórico de chat (render sempre o que já existe)
st.subheader("Conversa")
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada do usuário
user_input = st.chat_input("Digite sua pergunta (ex.: 'faça um histograma de Amount' ou 'quais conclusões até agora?').")

def push_assistant(text: str):
    st.session_state.chat_history.append({"role": "assistant", "content": text})
    with st.chat_message("assistant"):
        st.markdown(text)

if user_input:
    # 1) Adiciona e exibe imediatamente a pergunta do usuário
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # 2) Comandos de conclusões rápidas
    normalized = user_input.strip().lower()
    if normalized in {
        "conclusoes", "conclusões", "resumo", "insights",
        "quais conclusoes", "quais conclusões até agora?",
        "quais conclusões até agora"
    }:
        if st.session_state.insights:
            bullets = "\n".join([f"- {i}" for i in st.session_state.insights])
            push_assistant(f"Aqui estão as conclusões registradas até agora:\n\n{bullets}")
        else:
            push_assistant("Ainda não há conclusões registradas. Solicite alguma análise ou gráfico para começarmos.")
        st.stop()
    
    # 3) Classificação de intenção
    intent = classify_intent(user_input)
    if intent not in {"analysis", "chat"}:
        push_assistant("Não consegui entender a intenção. Reformule pedindo uma análise dos dados ou continue a conversa.")
        st.stop()
    
    # 4) Roteamento
    if intent == "chat":
        reply = get_chat_response(user_input)
        push_assistant(reply)
        st.stop()
    
    # intent == "analysis"
    df = st.session_state.df
    sample_text = df.head(20).to_markdown(index=False)
    code = get_analysis_code(user_input, sample_text)
    
    # --- Correção automática de linhas INSIGHT sem print() ---
    fixed_lines = []
    for line in code.splitlines():
        if line.strip().startswith("INSIGHT:"):
            insight_txt = line.strip().replace('"', "'")
            fixed_lines.append(f'print("{insight_txt}")')
        else:
            fixed_lines.append(line)
    code = "\n".join(fixed_lines)
    
    # Exibe o código gerado
    with st.expander("Código gerado pela IA", expanded=False):
        st.code(code, language="python")
    
    # Executa o código e captura saída/erros
    stdout_text, error_text = execute_code(code, df)
    
    if error_text:
        push_assistant(f"Ocorreu um erro na execução:\n\n``````")
        st.stop()
    
    # Exibe resultados textuais, se houver
    if stdout_text.strip():
        with st.chat_message("assistant"):
            st.markdown("Resultado da análise:")
            st.code(stdout_text)
        
        # Registra uma nota no histórico textual para indicar que houve saída/gráficos
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "Resultado da análise exibido e gráficos renderizados acima."
        })
        
        # CORREÇÃO APLICADA: Melhor extração e registro de INSIGHT
        new_insights = []
        for line in stdout_text.splitlines():
            line_stripped = line.strip()
            # Verifica se a linha contém INSIGHT (case-insensitive)
            if line_stripped.upper().startswith("INSIGHT:") or "INSIGHT:" in line_stripped.upper():
                # Extrai o texto após INSIGHT:
                if ":" in line_stripped:
                    insight = line_stripped.split(":", 1)[1].strip()
                    if insight:  # só adiciona se não está vazio
                        new_insights.append(insight)
                        # Debug: mostra que o insight foi encontrado
                        st.write(f"🔍 Debug: Insight encontrado: '{insight}'")
        
        # Adiciona os insights encontrados à memória
        if new_insights:
            st.session_state.insights.extend(new_insights)
            insight_count = len(new_insights)
            plural = "conclusão" if insight_count == 1 else "conclusões"
            push_assistant(f"✅ {insight_count} {plural} registrada(s) na memória!")
        else:
            # Debug: mostra quando nenhum insight foi encontrado
            st.write("🔍 Debug: Nenhum insight encontrado no output")
            # Ainda assim confirma execução
            push_assistant("Análise executada com sucesso.")