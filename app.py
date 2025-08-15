import os
import streamlit as st
from openai import OpenAI


# Initialize OpenAI client
client = OpenAI()


def get_default_system_prompt() -> str:
    return (
        "You are an expert senior software engineer and programming mentor. "
        "Provide accurate, deeply detailed, and production-grade answers to advanced programming "
        "questions across languages and frameworks. Prefer clear, concise explanations, discuss trade-offs, "
        "and include minimal, runnable code examples with comments. When relevant, propose tests, performance "
        "considerations, security implications, and alternatives. If the user’s goal is ambiguous, ask a brief "
        "clarifying question before answering."
    )


def build_messages(history, system_prompt):
    messages = [{"role": "system", "content": system_prompt.strip()}]
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    return messages


def call_openai_chat(messages, model="gpt-4", temperature=0.2, top_p=1.0, max_tokens=800):
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=float(temperature),
        top_p=float(top_p),
        max_tokens=int(max_tokens),
    )
    return response.choices[0].message.content


def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = get_default_system_prompt()
    if "model" not in st.session_state:
        st.session_state.model = "gpt-4"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.2
    if "top_p" not in st.session_state:
        st.session_state.top_p = 1.0
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 800


def sidebar():
    with st.sidebar:
        st.header("Settings")
        st.session_state.model = st.selectbox(
            "Model",
            options=["gpt-4", "gpt-3.5-turbo"],
            index=0 if st.session_state.get("model", "gpt-4") == "gpt-4" else 1,
            help="Choose the chat model.",
        )
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("temperature", 0.2)),
            step=0.05,
            help="Higher values = more creative; lower = more deterministic.",
        )
        st.session_state.top_p = st.slider(
            "Top-p (nucleus sampling)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("top_p", 1.0)),
            step=0.05,
        )
        st.session_state.max_tokens = st.slider(
            "Max tokens (response)",
            min_value=64,
            max_value=4096,
            value=int(st.session_state.get("max_tokens", 800)),
            step=32,
        )

        st.markdown("---")
        st.caption("System Prompt")
        st.session_state.system_prompt = st.text_area(
            "Role",
            value=st.session_state.get("system_prompt", get_default_system_prompt()),
            height=180,
            label_visibility="collapsed",
            help="Customize how the assistant behaves.",
        )

        st.markdown("---")
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.experimental_rerun()

        st.markdown("---")
        api_status = "Set" if os.environ.get("OPENAI_API_KEY") else "Not Found"
        st.caption(f"OPENAI_API_KEY: {api_status}")


def main():
    st.set_page_config(page_title="Advanced Programming Chatbot", page_icon="💻", layout="wide")
    st.title("💻 Advanced Programming Chatbot")
    st.caption("Ask advanced programming questions. The assistant provides production-grade, deeply detailed answers.")

    init_session_state()
    sidebar()

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask an advanced programming question...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    messages = build_messages(
                        st.session_state.messages,
                        st.session_state.system_prompt,
                    )
                    answer = call_openai_chat(
                        messages=messages,
                        model=st.session_state.model,
                        temperature=st.session_state.temperature,
                        top_p=st.session_state.top_p,
                        max_tokens=st.session_state.max_tokens,
                    )
                except Exception as e:
                    answer = f"Sorry, an error occurred while generating a response:\n\n{e}"

            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()