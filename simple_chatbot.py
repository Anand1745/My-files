# chatbot_agent.py

import openai

# STEP 1: Replace with your actual OpenAI API key
openai.api_key = "your-openai-api-key"

# STEP 2: Chatbot function
def chat_with_ai():
    print("AI Chatbot (type 'exit' to quit)")
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        
        messages.append({"role": "user", "content": user_input})
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages
        )
        
        reply = response.choices[0].message.content
        print("AI:", reply)
        
        messages.append({"role": "assistant", "content": reply})

# Run the chatbot
if __name__ == "__main__":
    chat_with_ai()
