import { SystemMessage } from "@langchain/core/messages";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { ChatGroq } from "@langchain/groq";
import readline from "readline";

const llm = new ChatGroq({
    apiKey:'',
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
})

// defining the prompt
const prompt = ChatPromptTemplate.fromMessages([
    ['system','you are a helpful assitant.'],
    new MessagesPlaceholder("chat_history"), // allows us to insert a list of messages ==> giving the llm our chat history to build a conversation
    ['human','{input}']
])

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

const chain = prompt.pipe(llm)

const chat_history = [];
async function chat() {
    rl.question("You: ", async (input) => {

        if (input.toLowerCase() === "exit") {
            console.log("chat ended , goodbye!");
            rl.close();
            return;
        }

        try {
            chat_history.push({ role: "user", content: input })
            const response = await chain.invoke({
                chat_history ,
                input
            });

            chat_history.push({ role: "assistant", content: response.content })
            console.log("AI:", response.content);

            // Continue the conversation
            chat();
        } catch (error) {
            console.error("Error:", error);
            rl.close();
        }
    });
}

console.log("Chat started! Type 'exit' to quit.");
chat();