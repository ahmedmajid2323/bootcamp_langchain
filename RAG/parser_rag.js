
import { Ollama, OllamaEmbeddings } from "@langchain/ollama";import { z } from "zod";
import { RunnableSequence } from "@langchain/core/runnables";
import { StructuredOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import dotenv from 'dotenv'
import path from 'path'
import { ChatGroq } from "@langchain/groq";
import { OutputFixingParser } from "langchain/output_parsers";
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768", 
    temperature: 0.5,        
});

// loading => define splitter => chunking => define embedder => save to db
// ....

  const zodSchema = z.object({
    credentials: z.object({
      name: z.string().describe("The name of the resume owner."),
      email: z.string().describe("The email of the resume owner."),
      phone_number: z.string().describe("The phone number of the resume owner."),
    }).describe('The credentials of the user\'s resume.'),
    projects: z.array(
      z.object({
        name: z.string().describe("The name of the project."),
        description: z.string().optional().describe("A brief description of the project."),
      })
    ).describe("A list of projects, each with a name and description."),
    certificates: z.array(
      z.object({
        name: z.string().describe("The name of the certificate."),
        description: z.string().optional().describe("A brief description of the certificate."),
      })
    ).optional().describe("A list of certificates, each with a name and description."),
  });

  const parser = StructuredOutputParser.fromZodSchema(zodSchema);

  const parserWithFix = OutputFixingParser.fromLLM(llm , parser)

  // Create a prompt template that instructs the model to return only the JSON object.
  // Note: We explicitly tell the model "do not include any extra text or markdown".
  const prompt = ChatPromptTemplate.fromTemplate(
    `Answer the user's question as best as possible using the context provided
    You must return only a JSON object that adheres exactly to the following schema (do not output any additional text or markdown):

    {format_instructions}

    context : {context}

    Question: {question}`
  );

  // Create the chain & invoke it 
  // ....
  