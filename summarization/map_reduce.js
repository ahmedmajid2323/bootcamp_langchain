import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { PromptTemplate } from "@langchain/core/prompts";
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";
import { pull } from "langchain/hub";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { ChatGroq } from "@langchain/groq";
import dotenv from 'dotenv'
import path from 'path'
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
})
  
const loader = new PDFLoader('./Hack_for_Hope.pdf')
const doc = await loader.load()
// just to return an array of texts
  
const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, 
    chunkOverlap: 200, 
});
  
// 3. Split the text into "Document" objects (LangChain's format)
const docs = await textSplitter.splitDocuments(doc);
// output : 
/* [
Document { pageContent: "First chunk of text..." },
Document { pageContent: "Second chunk of text..." },
// ...
] */

const map_prompt = await pull("rlm/map-prompt")
const reduce_prompt = await pull("rlm/reduce-prompt")

const chain = loadSummarizationChain(llm, {
  type: 'map_reduce', // refine , stuff
  map_prompt,
  reduce_prompt,
});

// 5. Run the chain to generate the final summary
const result = await chain.invoke({ 
    input_documents : docs
 });

console.log(result); 

// N.B : the map_reduce and refine methods are computationally expensive !! ==> solution : Clustering method