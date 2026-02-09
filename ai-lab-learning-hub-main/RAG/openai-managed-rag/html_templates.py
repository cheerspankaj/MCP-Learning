css = '''
    <style>
        body {
            font-family: 'Meta Pro', Arial, sans-serif;
            background-color: #F9F9F9;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
        }

        .chat-container {
            width: 90%;
            max-width: 600px;
            background-color: #FFFFFF;
            border-radius: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .chat-header {
            background-color: #009D94;
            color: #FFFFFF;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
            text-align: center;
        }

        .chat-messages {
            flex: 1;
            padding: 15px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        .chat-message {
            display: inline-block;
            border-radius: 20px;
            padding: 10px 15px;
            max-width: 70%;
            word-wrap: break-word;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px
        }

        .chat-message.bot {
            margin-left: auto;
            background-color: #FFFFFF;
            color: #009D94;
            text-align: left;
        }

        .chat-message.user {
            margin-right: auto;
            background-color: #009D94;
            color: #FFFFFF;
            text-align: left;
        }

        .chat-input {
            display: flex;
            padding: 10px;
            border-top: 1px solid #E0E0E0;
            background-color: #FFFFFF;
        }

        .chat-input textarea {
            flex: 1;
            padding: 10px;
            font-size: 14px;
            border: 1px solid #E0E0E0;
            border-radius: 15px;
            resize: none;
            height: 40px;
            outline: none;
        }

        .chat-input button {
            margin-left: 10px;
            padding: 10px 20px;
            font-size: 14px;
            color: #FFFFFF;
            background-color: #009D94;
            border: none;
            border-radius: 15px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }

        .chat-input button:hover {
            background-color: #007F7B;
        }
    </style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="message">{{MSG}}</div>
</div>
'''