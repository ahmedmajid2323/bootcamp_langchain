import { tool } from '@langchain/core/tools';
import { pull } from "langchain/hub";
import dotenv from 'dotenv';
import { AgentExecutor, createReactAgent } from 'langchain/agents';
import { BufferMemory } from 'langchain/memory';
import path from 'path';
import { ChatGroq } from '@langchain/groq';
import readline from 'readline';
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

// definig the tools ==> prompt engineering ==> define the agent (ll , tools , memory) ==> execute the agent

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768", 
    temperature: 0.5,        
  });

const get_time = async () =>{
    const now = new Date();
      return now.toLocaleTimeString("en-US", {
        hour: "numeric",
        minute: "2-digit",
        hour12: true,
      }); // retuuning current-time => HH:MM AM/PM
  }

  const currentTimeTool = tool(
    get_time ,
    {
      name: "current-time",
      description: "Useful when the user asks for the current time.",
      // returnDirect => retuning the tool's output directly without formatting
      // schema => defining the input schema for the tool's function
    }
  );

const greeting_user = tool(
    ()=>{
        return 'hello , i am a helpful AI agent, how can i assist you today ? '
    }
    ,
    {
        name: 'greetingUser' ,
        description : 'this tool is useful for greeting back the user' ,
    }
    
)  

const tools = [currentTimeTool , greeting_user];
const prompt = await pull("hwchase17/react") ; // hwchase17/react-chat , hwchase17/structured-chat-agent

const memory = new BufferMemory({
  memoryKey: "chat_history",
  returnMessages: true, // Return messages as BaseMessage objects
  inputKey: "input",
  outputKey: "output",
});

const agent = await createReactAgent({
    llm,
    tools,
    prompt,
  });

// we combine the agent (the brains) with the tools inside the AgentExecutor
// (which will repeatedly call the agent and execute tools)
const agentExecutor = new AgentExecutor({
    agent,
    tools,
    memory,
});

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

// Function to start the conversation
const startConversation = async () => {
  rl.question("You: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      console.log("Goodbye!");
      rl.close();
      return;
    }

    // Invoke the agent with the user's input
    const response = await agentExecutor.invoke({ input });
    console.log(response)

    startConversation();
  });
};

// Start the conversation
console.log("Type 'exit' to end the conversation.");
startConversation();