import { OpenAI } from 'langchain/llms/openai';
import { PineconeStore } from 'langchain/vectorstores/pinecone';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

const CONDENSE_PROMPT = `Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say you don't know. DO NOT try to make up an answer.
If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.

{context}

Question: {question}
Helpful answer in markdown:`;
const CHINESE_CONDENSE_PROMPT= `给定以下对话和一个后续问题，请重新表述后续问题成一个独立的问题。

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;
const CHINESE_QA_PROMPT = `你是一个乐于助人的AI助手。使用以下上下文片段来回答最后的问题。
如果你不知道答案，直接说不知道即可。请勿尝试编造答案。
如果问题与上下文无关，请礼貌地回答你只回答与上下文相关的问题。

{context}

Question: {question}
Helpful answer in markdown:`;
const CHINESE_ROLE_CONDENSE_PROMPT= `你的名字是{name},之后“Chat History”后跟的内容是你的记忆，“Follow Up Input”后跟的内容是另一个人和你的对话，请在“Standalone question”之后填上“Follow Up Input”后跟的内容并用括号添加{name}此时的想法。

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:`;
const CHINESE_ROLE_QA_PROMPT = `你的名字是{name}。使用以下上下文片段来填写“Question”之后的对话。
其中括号的内容是{name}此时的想法。
请把{name}的回复填在“Helpful answer in markdown”之后。

{context}

Question: {question}
Helpful answer in markdown:`;
const TOPK=4
const TEMPERATURE=0.5
export const makeChain = (vectorstore: PineconeStore) => {
  const model = new OpenAI({
    temperature: TEMPERATURE, // increase temepreature to get more creative answers
    modelName: 'gpt-3.5-turbo', //change this to gpt-4 if you have access
  });

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(TOPK),
    {
      qaTemplate: CHINESE_ROLE_QA_PROMPT.replace(new RegExp(/{name}/, 'g'),`汤姆`),
      questionGeneratorTemplate: CHINESE_ROLE_CONDENSE_PROMPT.replace(new RegExp(/{name}/, 'g'),`汤姆`),
      returnSourceDocuments: true, //The number of source documents returned is 4 by default
    },
  );
  return chain;
};