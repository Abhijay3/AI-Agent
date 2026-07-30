from agent_core import run_turn
from memory import load_history, save_history


def main():
    session_id = input("Session ID (e.g. your name): ").strip() or "default"
    messages = load_history(session_id)
    print("Chat started. Type 'exit' to quit.")
    if messages:
        print(f"(resumed conversation, {len(messages)} previous messages loaded)")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})
        reply = run_turn(messages)
        print(f"AI: {reply}")
        save_history(session_id, messages)


if __name__ == "__main__":
    main()
