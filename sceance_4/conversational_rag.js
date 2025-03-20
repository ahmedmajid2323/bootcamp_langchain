import { Chroma } from "@langchain/community/vectorstores/chroma";
import { OllamaEmbeddings } from "@langchain/ollama";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createRetrievalChain } from "langchain/chains/retrieval";
import readline from 'readline'
import dotenv from 'dotenv';
import path from 'path';
import { ChatGroq } from "@langchain/groq";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768", 
    temperature: 0.5,        
});
  
// Step 1: Generate embeddings and store them in Chroma
  const embeddings = new OllamaEmbeddings({
  model: "nomic-embed-text", 
});

const loader = new PDFLoader('file path here')
const docs = await loader.load()

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000 ,
  chunkOverlap: 50
})
const splitted_docs = await textSplitter.splitDocuments(docs)
console.log(`document splitted to ${splitted_docs.length} chunks`)

const vectorStore = await MemoryVectorStore.fromDocuments(splitted_docs , embeddings)

/* const vectorStore = new Chroma(embeddings, {
    url: "http://localhost:8000", 
    collectionName: "langsmith_doc", 
}); */

const retriever = vectorStore.asRetriever(); 

// this prompt helps llm that it should reformulate the Question of the user based on the chat history to make it stands for its own
// llm re-writes a stand alone quesry
// example ( user : how old is he when he died ? ) ==> reformulate based on chat_history ( llm : how old is King Arthur when he died ? )
const contextualizeQSystemPrompt =
"Given a chat history and the latest user question " +
"which might reference context in the chat history, " +
"formulate a standalone question which can be understood " +
"without the chat history. Do NOT answer the question, " +
"just reformulate it if needed and otherwise return it as is."

const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
  ["system", contextualizeQSystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

// this retriever rephrases the user's last input into the stand alone query
// and uses this query to **fetch** the most relevant docs from the VectorStore
const historyAwareRetriever = await createHistoryAwareRetriever({
  llm,
  retriever,
  rephrasePrompt: contextualizeQPrompt,
});

// defining the Q&A prompt to tell the LLM to answear the user's question based on the context provided (the one retrieved from historyAwareRetriever)
const QAsystemPrompt =
  "You are an assistant for question-answering tasks. " +
  "Use the following pieces of retrieved context to answer " +
  "the question. If you don't know the answer, say that you " +
  "don't know. Use three sentences maximum and keep the " +
  "answer concise." +
  "\n\n" +
  "{context}";

const qaPrompt = ChatPromptTemplate.fromMessages([
  ["system", QAsystemPrompt],
  new MessagesPlaceholder("chat_history"),
  ["human", "{input}"],
]);

// createStuffDocumentsChain ==> a chain that stuffs all retrieved doc into the qaPrompt
// it will take the retrieved doc from historyAwareRetriever and combine them into a single context : {context}
const questionAnswerChain = await createStuffDocumentsChain({
  llm,
  prompt: qaPrompt,
});

// defining the final chain historyAwareRetriever => questionAnswerChain 
const ragChain = await createRetrievalChain({
  retriever: historyAwareRetriever, // rephrase + fetch
  combineDocsChain: questionAnswerChain, // generate answear using retrieved docs
});

  console.log("Start chatting with the AI! Type 'exit' to end the conversation.");
  const chatHistory = []; // Collect chat history here (a sequence of messages)

  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  async function chat() {
  rl.question("You: ", async (input) => {
    if (input.toLowerCase() === "exit") {
      console.log("chat ended , goodbye!");
      rl.close();
      return;
    }

    try {
      chatHistory.push({ role: "user", content: input });
      const result = await ragChain.invoke({
        input,
        chat_history: chatHistory,
      });

      console.log(result.content)
      chatHistory.push({ role: "assistant", content: result.content });

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