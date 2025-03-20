import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { CheerioWebBaseLoader } from "@langchain/community/document_loaders/web/cheerio";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableSequence } from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { OllamaEmbeddings } from "@langchain/ollama";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import dotenv from 'dotenv'
import path from 'path'
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768", 
    temperature: 0.5,        
});
  
// Step 1: Generate embeddings and store them in Chroma
    const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", 
});

/* const loader = new PDFLoader('') */
const loader = new CheerioWebBaseLoader('https://en.wikipedia.org/wiki/Artificial_intelligence')
const docs = await loader.load() 

/* console.log(docs[0].pageContent) */

const textSplitter = new RecursiveCharacterTextSplitter({
  chunkSize: 1000 ,
  chunkOverlap: 50
})
const splitted_docs = await textSplitter.splitDocuments(docs)
console.log(`document splitted to ${splitted_docs.length} chunks`)

const vectorStore = await MemoryVectorStore.fromDocuments(splitted_docs , embeddings)

const retriever = vectorStore.asRetriever({
    k: 3 , // par defaut c 3 ; nb of relevent chunks retireved according to the user's query
});

const result = await retriever.invoke('what are the goals') // returns a list of Document objects that are most relevant to the query

//view le retrieved results
//....

const prompt = ChatPromptTemplate.fromTemplate(
    'answear to the user s question as best as you can : {question} , '+
    'with this provided {context} '
)

// defining the chain ( question + context ) ==> prompt => llm => parser
//....

//invoking the chain for final response
//....
