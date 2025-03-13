import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableLambda, RunnableSequence } from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import { CommaSeparatedListOutputParser } from "langchain/output_parsers";

const llm = new ChatGroq({
    apiKey:'',
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
}); 

//giving a meal idea for a list of ingredients

const prompt = ChatPromptTemplate.fromMessages([
    ['system','Answer the user\'s question by providing ONLY a comma-separated list of ingredients. Do not include any extra text, explanations, or formatting.'],
    ['human','{input}']
])

const parser = new CommaSeparatedListOutputParser() // this output parser returns a list

const guess_the_meal = new RunnableLambda({
    func : async (output)=>{
        const ingredients = output.join(',') // converting the list to a string
        console.log('the ingredients :', output)

        //here we are creating a chain to generate meals ideas for the provided ingredients
        const prompt = ChatPromptTemplate.fromTemplate(
            'given this list of ingredients {ingredients} , can you tell the possible meals i can prepare with it ?'
        )
        const chain = prompt.pipe(llm)
        const answear = await chain.invoke({ingredients})

        return answear.content
    }
})

// defini the extended chain ==> each output is the input of the next element !!
const chain = RunnableSequence.from([
    prompt,
    llm,
    parser,
    guess_the_meal
])

// finally invoking the chain to get the final result 
const result = await chain.invoke({input : 'give a list of ingredients for tunisan meal'})

console.log(result)
