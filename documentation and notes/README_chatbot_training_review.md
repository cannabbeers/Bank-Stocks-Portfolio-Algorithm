## Prompt Engineering Large Language Model For This Project


**Prompt Engineering** Definition:** *process of designing, testing, and refining prompts or questions that are given to a machine learning model, especially language models, to elicit desired responses or behaviors. The goal is to optimize the input to the model in a way that the output aligns with the intended result.*
*(per [ChatGPT 4](https://chat.openai.com/))*

1. Define the Objective: 

    Create a chatbot that has been given some current information about the bank sector, specific bank stocks, and other information that is current since the base GPT4 model is only trained up to September of 2021.  We utilize the ChatGPT API from **[OpenAI](https://openai.com/product)** and want to make sure it can provide recent financials with relevant numbers from 2023.

2. Find Relevant Data:

    There are many ways to collect data and prepare it to feed the model.  It can be done manually by typing out definitions, copying and uploading .csv and .pdf files, etc., and then writing `Python` code that would instill this information into the chatbot.  We needed a faster solution so we decided to collect the data directly from ChatGPT-4 by leveraging the built-in *Plugin store* to *Install* plugins that can fetch the information we needed to enhance the bot's functionality.  In this initial version we are not enabling plugins for our chatbot, however, we do utilize them to gather the data we want, manipulate and organize the data, and then decide what data we wish to load for the bot through another 3rd party application called **[Chatbase](https://www.chatbase.co/)**

3. Choose and Install Plugins for [ChatGPT 4](https://chat.openai.com/) *(to build our chatbot's data vocabulary and knowledge)*:

    The dropdown menu centered near the top of the [ChatGPT 4](https://chat.openai.com/) page provides access to the store and has a simple interface to browse and search for relevant plugins.  Navigate to the 'Plugin store' from there and easily find helpful third party plugins that connect ChatGPT to external apps and Chat GPT automatically chooses when to use during the conversation.  This process takes some level of trial and error as not all plugins are updated frequently and it's always a good idea to make sure you trust 3rd party apps before installing.  After a lot of trial and error we chose to utilize these 3 plugins to assist in building the data we wanted to feed our chatbot.
    + [Portfolio Pilot](https://portfoliopilot.com/) - *Your AI investing guide: portfolio assessment, recommendations, answers to all finance questions.*
    + [Statis Fund Finance](https://www.statisfund.com/about) - *Financial data tool for analyzing equities. You can get price quotes, analyze moving averages, RSI, and more.*
    + [Wolfram](https://products.wolframalpha.com/api)- *Access computation, math, curated knowledge & real-time data through Wolfram|Alpha and Wolfram Language.*

4. Collect Data and Organize Results Through Iterative Process:

    Starting with simple, basic inquiries about our subject matter, to ChatGPT-4 (with our chosen plugins enabled), the process quickly becomes a journey into a long conversation of asking for information, assesing what the responses mean, evaluating usefulness, and then back feeding the answers to inquires into the conversation to manipulate findings to generate organized and useful information without need for any code.  There is no one way to achieve this goal, but the steps taken here are right out of the playbook of a **Prompt Engineer**.
    1. start with a basic prompt - *"please list all major bank stocks and provide relevant financials for each"*
    2. iterative refinement - every response to our inquiries provides some information, copy and paste the responses back into conversation with more questions or instructions, here are some examples of just some of what was done to get the desired information that we will later load into our bot using [Chatbase](https://www.chatbase.co/):
        + *"list stock prices and important financial information about this list of bank stocks"* 
        + *"take this list and give me important financials, ratios, stock prices, and all details useful to investors"*
        + *" a lot of items have a topic yet say 'data not available', can you take the data you know and then reverse engineer missing statistics from formulas in `Wolfram` plugin?"*
        + *"organize and list the information in a report"*
    3. check the results for accuracy - one useful plugin that we initially used, but dropped called `Polygon` had a lot of good information to add, but its stock prices were over 6 months old when checked against actual current stock prices.  Based on this, we decided to drop the *Polygon* plugin while refining and manipulating the data to become better organized and avoid miscalculations when we asked for specific ratios and other useful information any investor may want.  [Statis Fund Finance](https://www.statisfund.com/about) currently has up to date stock prices and other financial data that proved invaluable to the process.
    4. manipulate and organize data to prepare all information we will load into our bot - now that we have verified true financial data, stock prices, etc. we continue to further refine, backtest, formulate, reorganize, output and gather more data.  there was no one single plugin that could just solve our needs here.  the 3 chosen plugins all added value in this process.

5. Actual Conversation Logs with Plugin assisted ChatGPT model (you may need to sign up for [OpenAI](https://openai.com/) to view content at the links below):
    Here are some of the links to view the actual conversations and how they unfolded to get the raw data, look for missing and inconsistent information, and then query ChatGPT to refine, fix, and review chats, enable or diasable plugins as needed while cross checking.  We pulled information from these logs to place into the training data for our chatbot.  While some time has been spent to be as accurate as possible on all items, there can still be confusion about the final numbers and what is selected, but overall most information appears to be based on accurate stock prices and latest financial data that is publicly reported as of the trade date of 8/23/2023.
    + [preliminary questions and hunt for useful plugins](https://chat.openai.com/share/7ce7ea02-553b-47c1-943b-e4d7840638c6)
    + [queries about macroeconomic data](https://chat.openai.com/share/7ce7ea02-553b-47c1-943b-e4d7840638c6)
    + [confirmed that Statis Fund Finance uses current stock price for its output](https://chat.openai.com/share/5b09cd35-46a1-4b4b-8c0c-37e9e26e3f7e)
    + [confirmed the 'Polygon' plugin did not have updated stock prices, even though GPT confidently stated they were accurate](https://chat.openai.com/share/d3c6c114-326c-4f61-b3bd-2a0443c2230e)
    + [questioning the findings of ChatGPT in a new chat](https://chat.openai.com/share/d3c6c114-326c-4f61-b3bd-2a0443c2230e)
    + [asking ChatGPT to be sure to use the prices from the Statis Fund Finance plugin to review and compare](https://chat.openai.com/share/38e3eeb0-5084-4f16-899b-8d1b1b66853d)
    + [more dialog with GPT-4](https://chat.openai.com/share/556bdec5-2561-41f5-a9aa-f14478240225)
    + [feeding back output from other chat back into a new chat to update, recalculate, and compare](https://chat.openai.com/share/c6066d95-9a9d-4cab-a847-84c0eb9338fb)
    + [asking for advice on returns, rebalancing portfolio, etc.](https://chat.openai.com/share/7ff1de54-2f2c-4ac3-b000-e6e183c7d9f8)
    + [slicing and dicing the results, asking for advice, hypothetical portfolio](https://chat.openai.com/share/e5ca29c6-3469-4b28-a20d-87ed841e73ec)
    + [building final reports and asking ChatGPT to fix its blind spots for 'data not available' when it previously delivered important information](https://chat.openai.com/share/3733544e-4bb9-46c0-a10f-435f52c08a47)

6. Copy and Paste: 

    Throughout the process we are just asking more questions, reviewing, organizing, verifying, and reordering information received in our conversations with ChatGPT-4.  When any inquiry to ChatGPT proved to meet usefulness criteria aforementioned here, it can simply be copied and pasted to another .doc or .txt file and just dropped directly into [Chatbase](https://www.chatbase.co/) to optimize the results for anyone using our chatbot.
