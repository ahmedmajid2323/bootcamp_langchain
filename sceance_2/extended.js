import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableSequence } from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { CommaSeparatedListOutputParser } from "langchain/output_parsers";
import dotenv from 'dotenv'
import path from 'path'
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
}); 

//giving a meal idea for a list of ingredients

const prompt = ChatPromptTemplate.fromMessages([
    ['system','Answer the user\'s question by providing ONLY a comma-separated list of ingredients. Do not include any extra text, explanations, or formatting.'],
    ['human','{input}']
])

const parser = new CommaSeparatedListOutputParser() // this output parser returns a list

// defining the extended chain ==> each output is the input of the next element !!
const chain = RunnableSequence.from([
    prompt,
    llm,
    parser,
    // add a fucntion (chain) that generates meals ideas for the provided ingredients
])

// finally invoking the chain to get the final result 
const result = await chain.invoke({input : 'give a list of ingredients for tunisian meal'})

console.log(result)
