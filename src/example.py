from emotion_chat_bot.EmotionChatBot import EmotionChatBot

# 情緒傾向（會依此生成 Ideal Emotion Representation）
# emotion_tendency: dict = {
# 	"neutral": 0.2,
# 	"anger": 0.4,
# 	"disgust": 0.5,
# 	"fear": 0.1,
# 	"happiness": 0.8,
# 	"sadness": 0.2,
# 	"surprise": 0.2,
# }
# 預設是隨機產生
emotion_tendency = None

bot = EmotionChatBot(emotion_tendency=emotion_tendency)

print("\n\n\n")
print("If you want to end the conversation, you can enter 'quit'.")
while True:
	user_message: str = input("User: ").strip()
	if user_message == "quit":
		print("Bot: Goodbye!")
		break

	result = bot(user_message)

	print(f"Bot({result['emotion']}): {result['response']}")

emotions: list = ["neutral", "anger", "disgust", "fear", "happiness", "sadness", "surprise"]

emotion_representation: list = bot.similarity_analyzer.ideal_emotion_representation.tolist()

print(f"Bot's emotion representation:\n{dict(zip(emotions, emotion_representation))}")
