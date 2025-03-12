import { AIMessage, HumanMessage, SystemMessage } from '@langchain/core/messages';
import { ChatPromptTemplate } from '@langchain/core/prompts';
import { RunnableSequence } from '@langchain/core/runnables';
import { ChatGroq } from '@langchain/groq';
/* import dotenv from 'dotenv';

dotenv.config();

console.log("Loaded ENV:", process.env.GROQ_API_KEY); */

const llm = new ChatGroq({
    apiKey:'',
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
}); 

// ' .fromMessages ' takes an array of message arrays , defining 3 different roles (system , assistant , user) 
const prompt_template_1 = ChatPromptTemplate.fromMessages([
    ['system','you are a helpful assistant.'],
    ['user',"{user_input}"]
    /* new SystemMessage("You are a helpful assistant."), // same thing works with with this syntax
    new HumanMessage("Hi, can you help me with something?"),
    new AIMessage("Of course! What do you need help with?"),
    new HumanMessage("Tell me a joke about {topic}."), */
])

// ' .fromTemplate ' takes one string , telling the llm what to do
const prompt_template_2 = ChatPromptTemplate.fromTemplate(
    'you are a helpful assistant is sports, tell me about {sport} '
)

// here we are creating the chain, which combines the two elements : llm (to generate response) & prompt 
const chain_1 = prompt_template_1.pipe(llm)
const chain_2 = prompt_template_2.pipe(llm)

// another method to declare chains
/* const chain_1 = RunnableSequence.from([
    prompt_template_1,
    llm
]) */

// getting the final response
const result_1 = await chain_1.invoke({
    user_input: 'who are you ?' ,
})
const result_2 = await chain_2.invoke({
    sport: ' tennis ' ,
})

