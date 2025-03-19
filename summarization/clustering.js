import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { ChatGroq } from "@langchain/groq";
import { OllamaEmbeddings } from "@langchain/ollama";
import { loadSummarizationChain } from "langchain/chains";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import {kmeans} from "ml-kmeans";
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
  chunkSize: 500, 
  chunkOverlap: 100, 
});

// 3. Split the text into "Document" objects (LangChain's format)
const docs = await textSplitter.splitDocuments(doc);
console.log(`document splitted to ${docs.length} chunks`)
// output : 
/* [
  Document { pageContent: "First chunk of text..." },
  Document { pageContent: "Second chunk of text..." },
  // ...
] */

  const embeddings = new OllamaEmbeddings({
    model: "nomic-embed-text", 
  });

  const embeddingsList = await embeddings.embedDocuments(
    docs.map((doc) => doc.pageContent)
  );

  const results = kmeans(embeddingsList, 3);
  const clusters = results.clusters

  console.log(clusters) // an array indicating the categories of each chunk after grouping

  // converting the embeddings to texts after grouping the similar chunks
  const clusteredDocs = clusters.reduce((acc, clusterIndex, i) => {
    if (!acc[clusterIndex]) acc[clusterIndex] = [];
    acc[clusterIndex].push(docs[i]);
    return acc;
  }, []);

  console.log(clusteredDocs)

  const summarizeCluster = async (clusterDocs) => {
    const chain = loadSummarizationChain(llm, {
      type: "stuff", // Use the "stuff" method for each cluster ,
    });
  
    const result = await chain.invoke({
      input_documents: clusterDocs,
    });
  
    return result.text;
  };

  const clusterSummaries = await Promise.all( //summarize clusters in parallel, which can significantly speed up the process.
    clusteredDocs.map((docs) => summarizeCluster(docs))
  );
  
  // 7. Combine cluster summaries into a final summary
  const finalSummary = clusterSummaries.join("\n\n");

  console.log(finalSummary)