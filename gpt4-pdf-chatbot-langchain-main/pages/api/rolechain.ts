import { makeChain } from '@/utils/makechain';
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

对话历史:
{chat_history}
后续输入: {question}
独立问题:`;
const CHINESE_QA_PROMPT = `你是一个乐于助人的AI助手。使用以下上下文片段来回答最后的问题。
如果你不知道答案，直接说不知道即可。请勿尝试编造答案。
如果问题与上下文无关，请礼貌地回答你只回答与上下文相关的问题。

{context}

问题: {question}
有用的回答（使用Markdown格式）:`;
const CHINESE_ROLE_CONDENSE_PROMPT= `请总结对话历史，并根据总结的对话历史给出形如"后续输入(对话总结)"修饰。请尽可能确保回复的形式是"后续输入(对话总结)"的状态，不要进行多余的回复。

对话历史:
{chat_history}
后续输入: {question}
后续输入(对话总结):`;
const CHINESE_ROLE_QA_PROMPT = `请根据下文提供的对话参与者的记忆以及对话后跟括号内的情景总结回复这个对话，你的身份不重要，仅仅需要结合记忆与情景推论对话者的回复即可。
不要进行多余的回复。

记忆：
{context}

对话: {question}
合理的回答（使用Markdown格式）:`;
export const roleChain = (vectorstore: PineconeStore,qaTemplate:string= QA_PROMPT,questionGeneratorTemplate:string= CONDENSE_PROMPT) => {
    const chain = makeChain(vectorstore,4,0.5,qaTemplate,questionGeneratorTemplate);
    return chain;
}
