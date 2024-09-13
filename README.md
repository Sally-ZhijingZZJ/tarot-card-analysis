# tarot-card-analysis
This repository contains two simple models I created(will be adding more models related to tarot cards in the future when i got time) for tarot card analysis.<br>
-TEXT CLASSIFICATION MODEL:<br> Takes in a one sentence question and output a spread that can best answer the question.<br>
The dataset is generated with chatgpt-4o(since it's hard to get enough data from real human analysis). It was generated by providing chatgpt with information on each spread, and generate questions that are close related/can be solved by the spread.(spread info from https://labyrinthos.co/blogs/learn-tarot-with-labyrinthos-academy) <br>
The given question can be classified into 8 possible spread：5 Card Tarot Spread for Decision Making, 10 card Tarot Spread for Self Growth and Personal Development, Repeating Tarot Card Spread, Are You Ready For Love Tarot Spread, Chakras Love Tarot Spread, New Years arot Spread, Finding Love Relationship Tarot, Facing Challenges Career Tarot Spread.<br>
<br>
One model is created using sklearn learnSVC, the other one is created with torch with a LSTM layer.<br>
sklearn EX: <br>
-input=what will my life behave next year?<br>
output=New Years Tarot Spread<br>
<br>
-input=how to resolve a challenge I'm facing in a relationship?<br>
output=Chakras Love Tarot Spread<br>
<br>
torch EX:<br>
-input=how to improve my career?<br>
-output=Facing Challenges Career Tarot Spread<br>
<br>
Note: since it was my first time tried to generate data with chatgpt, the dataset does not use a large variety of words, which cause the model to beave poorly with words that does not appeared in the dataset.(e.g. if input text contains "love", it will most likely classify the text into correct class, but if "boyfriend"/"girlfriend" are appeared in a question instead of "love", the model is likely to classify that into a wrong class.)<br>
Somehow the torch model perform poorly compare to the sklearn one, maybe bacause I didn't set up things properly for this model.
<br>
<br>
-TEXT GENERATION MODEL:<br>
This model I hadn't had figured how to build this model yet, but ideally it's going to take in card name, whether it's upright or reversed, and the location it lies in a spread, and generate an analysis baased on that. 
