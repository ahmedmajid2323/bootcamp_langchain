import { StructuredOutputParser , JsonOutputParser , StringOutputParser , CommaSeparatedListOutputParser} from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatGroq } from '@langchain/groq';
import dotenv from 'dotenv'
import { OutputFixingParser } from "langchain/output_parsers";
import path from 'path'
import { z } from "zod";
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
}); 

// **1 : stringOutputParser => It ensures that the output is treated as a string (useful in chains)
const str_parser = new StringOutputParser()

const str_prompt = ChatPromptTemplate.fromMessages([
    ["assistant",'you are a very funny guy, generate a joke oaver the user s input '],
    ["user"," {user_input} "]
])

const str_chain = str_prompt.pipe(llm).pipe(str_parser)
const str_res = await str_chain.invoke({user_input : 'pirate'})


// **2 : CommaSeparatedListOutputParser => Parses the output into a comma-separated list: ["flour", "sugar", "eggs"]
const list_parser = new CommaSeparatedListOutputParser()

const list_prompt = ChatPromptTemplate.fromMessages([
    ["assistant",'answear to the user s question in form of Comma Separated List , don t add any additional text or formatting '],
    ["user"," {user_input} "]
])

const list_chain = list_prompt.pipe(llm).pipe(list_parser)

const res = await list_chain.invoke({user_input : 'what are the ingredients to make a cake ?'})


// **3 : JsonOutputParser => Parses the output into a valid JSON object.
const json_parser = new JsonOutputParser()

const json_prompt = ChatPromptTemplate.fromTemplate(
    "Answer the user query."+
    "Respond with a valid JSON object, containing two fields: 'setup' and 'punchline' :\n {query}"
  );

const json_chain = json_prompt.pipe(llm).pipe(json_parser)

const response = await json_chain.invoke({query : 'Tell me a joke about software engineers.'})


// **4 : StructuredOutputParser => Enforces a specific schema or structure on the output.
const schema = z.object({
    origin : z.string().describe('the origin of the meal, where it comes from') ,
    recipe : z.array(z.string()).describe('the necessary ingrediants for the meal') ,
})

const parser = StructuredOutputParser.fromZodSchema(schema)

const prompt = ChatPromptTemplate.fromTemplate(
    'answear to the user s question as best as you can using this format instruction : \n {format_instruction} \n {user_input} '
)

const chain = prompt.pipe(llm).pipe(parser)

const result = await chain.invoke({
    user_input : 'bruscheta' ,
    format_instruction : parser.getFormatInstructions() ,
})

console.log(result)

