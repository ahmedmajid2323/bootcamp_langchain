import { RunnableLambda, RunnableParallel } from "@langchain/core/runnables";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatGroq } from "@langchain/groq";
import { CommaSeparatedListOutputParser, StringOutputParser } from "@langchain/core/output_parsers";
import dotenv from 'dotenv'
import path from 'path'
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
}); 

const prompt_template = ChatPromptTemplate.fromMessages([
    ["system", "You are an expert product reviewer"],
    ["user", "List the main features of this product: {product_name}"],
]);

const parser = new StringOutputParser();

/* Fix: Return the formatted prompt as a string , the error here was that analyse_prons and analyse_cons functions
should return the result of formatting the prompt, not the prompt template itself (CHECK DOCUMENTTIONS)  */
const analyse_prons = new RunnableLambda({
    func: async (features) => {
        const pros_template = ChatPromptTemplate.fromMessages([
            ["system", "You are an expert product reviewer"],
            ["user", "Given these {features}, list the pros of these features by providing ONLY a comma-separated list. Do not include any extra text, explanations, or formatting."],
        ]);
        return await pros_template.format({ features }); //return the formatted prompt as a string not ChatPromptTemplate object
    },
});

// Fix: Return the formatted prompt as a string
const analyse_cons = new RunnableLambda({
    func: async (features) => {
        const cons_template = ChatPromptTemplate.fromMessages([
            ["system", "You are an expert product reviewer"],
            ["user", "Given these {features}, list the cons of these features by providing ONLY a comma-separated list. Do not include any extra text, explanations, or formatting."],
        ]);
        return await cons_template.format({ features });
    },
});

const list_parser = new CommaSeparatedListOutputParser()

const prons_chain = analyse_prons.pipe(llm).pipe(list_parser);
const cons_chain = analyse_cons.pipe(llm).pipe(list_parser);

const chain = prompt_template
    .pipe(llm)
    .pipe(parser) // this parser will return the features that will be inserted in the RunnableLambda analyse_cons & analyse_prons
    .pipe(RunnableParallel.from({  // Executes multiple tasks in parallel, reducing latency
        prons: prons_chain,        // returns an object of cons and prons 
        cons: cons_chain 
    }))
    // we can chain another component to recommand us either we buy the product or no 

const run = async () => {
    try {
        const response = await chain.invoke({ product_name: "iPhone 13" });
        console.log("Final Response:", response);
    } catch (error) {
        console.error("Error during chain execution:", error);
    }
};

run();

 