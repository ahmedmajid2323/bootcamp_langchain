import { AgentExecutor, createReactAgent } from "langchain/agents";
import { pull } from "langchain/hub";
import { tool } from "@langchain/core/tools";
import { BufferMemory } from "langchain/memory";
import readline from 'readline';
import { z } from "zod";
import dotenv from 'dotenv';
import path from 'path';
import { ChatGroq } from "@langchain/groq";
import { OllamaEmbeddings } from "@langchain/ollama";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768", 
    temperature: 0.5,        
});

// create the conversational RAG tool
// create the health detector tool (sceance_3)

const tools = [];
const prompt = await pull("hwchase17/react-chat") // try hwchase17/react-chat ;  you need chat_history

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
const agentExecutor = AgentExecutor.fromAgentAndTools({
    agent,
    tools,
    memory,
    verbose: true ,
    returnIntermediateSteps: true ,
    handleParsingErrors: true,
});

// Create readline interface for terminal input/output
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
    console.log(response.output)
    console.log(response.intermediateSteps[0]?.action)

    startConversation();
  });
};

// Start the conversation
console.log("Type 'exit' to end the conversation.");
startConversation();