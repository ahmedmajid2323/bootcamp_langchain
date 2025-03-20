import { HumanMessage, SystemMessage } from "@langchain/core/messages";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { RunnableBranch, RunnableSequence } from "@langchain/core/runnables";
import { ChatGroq } from "@langchain/groq";
import dotenv from 'dotenv'
import path from 'path'
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: "mixtral-8x7b-32768",
    temperature: 0.7,
}); 

// the purpose of this prompt is returning a classification of the user's input 
const classification_prompt = ChatPromptTemplate.fromTemplate(
    "You are a health support assistant. Your task is to classify the user\'s message into one of the following categories:"+
    "1. **Mental Health **: either he is depressed , anxious , depressed , etc .. "+
    "4. **Physical Health - Feeling Sick**: The user describes physical symptoms like pain, fever, nausea, or fatigue."+
    "5. **Others**: The user's message does not fit into any of the above categories."+

    "**User Message**: {user_input}"+

    "**Instructions**:"+
    "- Respond with ONLY the category name (e.g., Mental Health )."+
    "- Do not provide any additional explanation or commentary."
)

const classification_chain = classification_prompt.pipe(llm).pipe(new StringOutputParser)

const prompt_mental_health = ChatPromptTemplate.fromTemplate(
    'You are a compassionate mental health assistant. The user is sharing something about their mental or emotional well-being.'+
    'Your task is to provide a supportive, empathetic, and non-judgmental response. Focus on listening, validating their feelings, and offering gentle suggestions for self-care.'+
    'Encourage them to seek professional help if needed, but avoid giving specific medical advice or diagnoses.'+

    '**User Message**: "{user_input}"'
)

const prompt_physical_health = ChatPromptTemplate.fromTemplate(
    'You are a health assistant. The user is feeling sick and describing physical symptoms. Your task is to provide general advice and encourage them to seek medical attention if necessary.'+

    "**User Message**: {user_input}"+

    "**Instructions**:"+
    "- Acknowledge their symptoms and express concern."+
    "- Suggest general self-care measures (e.g., rest, hydration, over-the-counter remedies if appropriate)."+
    "- Advise them to monitor their symptoms and seek medical attention if they worsen or persist."+
    "- Avoid giving specific medical diagnoses or treatments."
)

const other_prompt = ChatPromptTemplate.fromTemplate(
    "You are a helpful assistant. The user's message does not fall into the mental or physical health categories."+
    "Your task is to provide a kind and supportive response while gently guiding them to the appropriate resources if needed."+

    "**User Message**: {user_input}"
)

// defining chain for each class
const mental_health_chain = prompt_mental_health.pipe(llm)
const physical_health_chain = prompt_physical_health.pipe(llm)
const other_chain = other_prompt.pipe(llm)

//defining the branching ==> introducing different categories to handle
const branch = RunnableBranch.from([
    [
        (output) => output.class.toLowerCase().includes('mental health') ,
        mental_health_chain
    ],
    [
        (output) => output.class.toLowerCase().includes('physical health') ,
        physical_health_chain
    ],
    other_chain
])

// defining the extended chain
const final_chain = RunnableSequence.from([
    {
        class: classification_chain,
        user_input: (input) => input.user_input,
    },
    branch // this branch will take as input the object {class: '..' , user_input:'..'}
])

// invoking 'user_input' that will be inserted in classification_chain
const response = await final_chain.invoke({user_input : 'I\'ve been feeling really down lately and can\'t seem to enjoy anything.'})

console.log(response.content)