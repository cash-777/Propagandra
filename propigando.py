import joblib

model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

new_text = ["Aug 30 (Reuters) - President Volodymyr Zelenskiy dismissed Ukraine's Air Force Commander Mykola Oleshchuk on Friday, according to a presidential decree.The dismissal was announced just a day after the Ukrainian military reported that an F-16 jet crashed and its pilot died while repelling a major Russian strike on Monday.I have decided to replace the commander of the Air Forces... I am eternally grateful to all our military pilots, President Volodymyr Zelenskiy said in his evening address.He did not give a reason for dismissal but mentioned that personnel must be protected, and that there was a need to strengthen the command level.Ukraine's General Staff said that General Lieutenant Anatoliy Kryvonozhka would temporarily perform the duties of commander.The Ukrainian military did not provide a reason for Monday's crash but said the jet came down while it was approaching a Russian target. Oleshchuk said on Monday partners from the U.S. were helping to investigate the incident.A U.S. defense official told Reuters that the crash did not appear to be the result of Russian fire, and possible causes from pilot error to mechanical failure were still being investigated."]


new_text_tfidf = vectorizer.transform(new_text)

prediction = model.predict(new_text_tfidf)

if prediction[0] == 0:
    print("The text is classified as misinformation (Class 0).")
else:
    print("The text is classified as non-misinformation (Class 1).")
