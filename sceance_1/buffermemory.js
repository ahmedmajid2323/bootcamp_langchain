import { ChatPromptTemplate, MessagesPlaceholder } from '@langchain/core/prompts';
import { ChatGroq } from '@langchain/groq';
import { BufferMemory , ConversationSummaryMemory , ConversationSummaryBufferMemory } from 'langchain/memory';
import { ConversationChain } from 'langchain/chains';
import readline from 'readline';
import dotenv from 'dotenv'
import path from 'path'
dotenv.config({ path: path.resolve(process.cwd(), '../.env') });

const llm = new ChatGroq({
    model: 'mixtral-8x7b-32768',
    temperature: 0.7,
});

// a memory built in langchain that stores the chat_history in memory ==> buffer = temporary 
// memory stored in RAM (can be delete prev cache with memory.delete()) , but it can handle its own
const memory = new BufferMemory({
    returnMessages: true, 
    memoryKey: 'chat_history', 
});

// this type of memory Continually summarizes the conversation history ==> returns a summary of the chat history
/* const memory = new ConversationSummaryMemory({ 
    llm,
    returnMessages: true, 
    memoryKey: 'chat_history', 

}); */

// combines (bufferMemory & ConversationSummaryMemory) = stores the chat history in memory during runtime + maintains a summary of the conversation
/* const memory = new ConversationSummaryBufferMemory({
    llm,
    maxTokenLimit: 300 , // When the buffer exceeds this limit, the oldest messages are removed to make room for new ones.
    returnMessages: true, 
    memoryKey: 'chat_history', 
}); */

const prompt = ChatPromptTemplate.fromMessages([
    ['system', 'You are a helpful assistant.'],
    new MessagesPlaceholder('chat_history'), // Insert chat history here
    ['human', '{input}'], 
]);

const chain = new ConversationChain({
    llm,
    memory, // this type of chain, inserts automatically the chat_history in the prompt defined (inserts it in the messagesPlaceholder)
    prompt, 
});

const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
});

async function chat() {
    rl.question('You: ', async (input) => {
        if (input.toLowerCase() === 'exit') {
            console.log('Chat ended. Goodbye!');
            console.log(memory)
            rl.close();
            return;
        }

        try {
            const response = await chain.invoke({
                input: input,
            });

            console.log('AI:', response.response);

            chat();
        } catch (error) {
            console.error('Error:', error);
            rl.close();
        }
    });
}

console.log('Chat started! Type "exit" to quit.');
chat();