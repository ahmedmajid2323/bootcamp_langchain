import { tool } from '@langchain/core/tools';
import { pull } from "langchain/hub";
import dotenv from 'dotenv';
import { AgentExecutor, createReactAgent, createStructuredChatAgent } from 'langchain/agents';
import { BufferMemory } from 'langchain/memory';
import path from 'path';
import { ChatGroq } from '@langchain/groq';
import readline from 'readline';
import { z } from 'zod';
import { RunnableBranch, RunnableSequence } from '@langchain/core/runnables';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { StringOutputParser } from '@langchain/core/output_parsers';
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

// definig the tools ==> prompt engineering ==> define the agent (llm , tools , memory) ==> execute the agent

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768", 
    temperature: 0.5,        
});

// create the weather tool

const tools = [];
const prompt = await pull("hwchase17/structured-chat-agent") ;

const memory = new BufferMemory({
  memoryKey: "chat_history",
  returnMessages: true, // Return messages as BaseMessage objects
  inputKey: "input",
  outputKey: "output",
});

const agent = await createStructuredChatAgent({
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
    verbose: true ,
    returnIntermediateSteps: true
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
    console.log(response.intermediateSteps[0]?.action)

    startConversation();
  });
};

// Start the conversation
console.log("Type 'exit' to end the conversation.");
startConversation();