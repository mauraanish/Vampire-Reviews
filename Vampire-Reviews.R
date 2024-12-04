# import libraries
library(tidyverse) # for general coding
library(quanteda) # for text analysis
library(quanteda.textplots) # for visualizing text analysis results
library(quanteda.textmodels) # for visualizing text analysis results
library(caret) # for machine learning
library(e1071) # for machine learning
library(randomForest) # for machine learning

### Introduction

### Research Question and Hypotheses
# RQ: According to critics, what makes a good vampire movie? What words are commonly used to describe positively-reviewed vampire movies?
# H1: Some words that are used to positively describe any movie (such as "good," "amazing," and "fantastic") will be used to describe well-received vampire movies.
# H2: Other words that may not typically describe positive reactions to movies in general (such as "bloody," "gruesome," and "disturbing") may also be used to describe 
# well-received vampire movies.

### Loading and Cleaning Data
# On Kaggle (<https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/data>), I found a dataset consisting of over 1 million 
# reviews of over 17,000 movies from critics on Rotten Tomatoes. The data was scraped from the Rotten Tomatoes website on October 31, 2020, and thus only contains 
# information about movies released in 2020 or earlier.
# After cross-referencing the list of movies reviewed by critics in this dataset with a list of vampire movies, I narrowed down a final list of 154 vampire movie titles 
# contained in the dataset. By only retaining the reviews of vampire movies, I narrowed down the number of reviews from over 1 million to just over 9,000.

# read in dataset with only vampire movie reviews
vamp_reviews <- read.csv("vamp_reviews.csv")
vamp_reviews <- vamp_reviews |>
  rename(movie_ID = rotten_tomatoes_link) |>
  rename(review_freshness = review_type) |>
  select(movie_ID, critic_name, publisher_name, review_freshness, review_date, review_content) |>
  mutate(review_date = mdy(review_date)) |>
  arrange(review_date)
dim(vamp_reviews)
head(vamp_reviews)

# The number of reviews of vampire movies was further reduced after removing reviews from critics that only contained a rating with no additional text, reviews that 
# appeared in the dataset more than once, and reviews that weren't in English.

# remove rows with no review
vamp_reviews <- filter(vamp_reviews, str_length(review_content)>1)
dim(vamp_reviews)

# remove duplicate rows
vamp_reviews <- distinct(vamp_reviews)
dim(vamp_reviews)

# remove reviews that aren't in english
vamp_reviews <- vamp_reviews |>
  filter(publisher_name != "Cinema em Cena" & 
           publisher_name != "Cinenganos" &
           publisher_name != "EnPrimeur.ca" &
           publisher_name != "Movies for the Masses" &
           publisher_name != "Moviola" & 
           publisher_name != "Panorama" &
           publisher_name != "Uruguay Total")
dim(vamp_reviews)

### Exploratory Data Analysis
# After the reviews had been cleaned, I split them into two subsets: one consisting only of "Fresh" reviews (if the critic liked the movie), and one consisting only of 
# "Rotten" reviews (if the critic didn't like the movie). There was an almost equal number of Fresh (4,181) and Rotten (4,073) reviews.

# create fresh and rotten datasets
fresh_reviews <- filter(vamp_reviews, review_freshness == "Fresh")
dim(fresh_reviews)
rotten_reviews <- filter(vamp_reviews, review_freshness == "Rotten")
dim(rotten_reviews)

# create fresh corpus
fresh_reviews$review_ID <- str_c("F", seq(1:nrow(fresh_reviews)))
fresh_corpus <- corpus(fresh_reviews, docid_field = "review_ID",
                       text_field = "review_content")
summary(fresh_corpus, n=10)

# create rotten corpus
rotten_reviews$review_ID <- str_c("R", seq(1:nrow(rotten_reviews)))
rotten_corpus <- corpus(rotten_reviews, docid_field = "review_ID",
                       text_field = "review_content")
summary(rotten_corpus, n=10)

#### Word Clouds
# Next, I wanted to get a sense of which words were most frequently used in both corpora.

# tokenize; remove punctuation, numbers, and stopwords
fresh_tokens <- tokens(fresh_corpus, 
                       remove_punct = T, remove_numbers = T) |>
  tokens_select(pattern=c("film", "films", "movie", "movies", 
                          "vampire", "vampires", "one", "two",
                          "like", "just", "still", "much", "even"), 
                select = "remove")

# make document feature matrix
fresh_dfm <- dfm(fresh_tokens) |>
  dfm_remove(stopwords('english')) |>
  dfm_trim(min_termfreq = 50, verbose = FALSE)

# make word cloud
set.seed(12)
textplot_wordcloud(fresh_dfm, color="darkgreen")

# tokenize; remove punctuation, numbers, and stopwords
rotten_tokens <- tokens(rotten_corpus, 
                       remove_punct = T, remove_numbers = T) |>
  tokens_select(pattern=c("film", "films", "movie", "movies", 
                          "vampire", "vampires", "one", "two",
                          "like", "just", "still", "much", "even"), 
                select = "remove")

# make document feature matrix
rotten_dfm <- dfm(rotten_tokens) |>
  dfm_remove(stopwords('english')) |>
  dfm_trim(min_termfreq = 50, verbose = FALSE)

# make word cloud
set.seed(12)
textplot_wordcloud(rotten_dfm, color="red")

# The word clouds for the positive and negatively reviewed vampire movies have some overlap, but they also have their differences. The Fresh word cloud has words like 
# "good," "great," "best," "fun," "original," "love," and "entertaining," which are expected to be associated with positive reviews. The Rotten word cloud has words like
# "bad," "enough," "little," "never," "worst," "nothing," and "mess," which are expected to be associated with negative reviews. However, the Rotten word cloud also 
# contains the common positive words like "good," "great," "best," "fun," "original," and "love," which might be frequently negated in the negative reviews, but that isn't
# captured by the unigram analysis performed here.
# Both word clouds also contain some words related to specific movies, such as "Twilight," "Underworld," "Dracula," "Blade," and "Bella." It's not surprising that the vampire
# movie franchises with multiple entries like *Twilight* (5 films), *Underworld* (5 films), and *Blade* (3 films) are discussed more than films which stand alone. The 
# character of Dracula is also featured in many different vampire movies, and even if he isn't explicitly present in them, he presents a well-known figure for critics to 
# compare the depictions of other vampires to in their reviews. Other vampire-specific words in the clouds include "horror," "action," "blood," "dark," "shadows," "gothic,"
# and "scary."
# Finally, both word clouds feature words pertaining to aspects of the movies that critics discuss. These words include "story," "plot," "characters," "genre," "effects," 
# "script," "director," and "performance." Going forward, it could be helpful to study how these elements of the movies are discussed (such as which adjectives frequently
# accompany them) to distinguish positive from negative reviews. A similar tactic could also be applied to the adjectives used near common words like "film," "movie," or 
# "vampire."

# make a corpus with 2 documents (1 for all fresh and 1 for all rotten reviews)
fresh_review_text <- paste(fresh_reviews$review_content, collapse=" ")
rotten_review_text <- paste(rotten_reviews$review_content, collapse=" ")
fr_df <- tibble(freshness = c("Fresh", "Rotten"),
                review_text = c(fresh_review_text, rotten_review_text))
fr_corpus <- corpus(fr_df, docid_field = "freshness", text_field = "review_text")

# tokenize; remove punctuation, numbers, and stopwords
fr_tokens <- tokens(fr_corpus, remove_punct = T, remove_numbers = T) |>
  tokens_select(pattern="vampire", select = "remove")

# make document feature matrix
fr_dfm <- dfm(fr_tokens) |> dfm_remove(stopwords('english')) |>
  dfm_trim(min_termfreq = 100, verbose = FALSE)

# make comparison word cloud
set.seed(15)
textplot_wordcloud(fr_dfm, comparison = TRUE, color = c("darkgreen", "red"))

# The comparison word cloud highlights many of the differences that are observable from the separate word clouds above. However, this word cloud makes it clear that 
# the Fresh reviews are more likely to contain positive words like "great," "best," "original," "fun," "entertaining," and "classic" than Rotten reviews are. Rotten 
# reviews are more likely to contain negative words like "bad" and "nothing" along with words about largely negatively-received series such as "Twilight," "Underworld,"
# "Blade," and the word "franchise" in general.

#### Co-Occurrence Networks
# I was curious to see if the ways in which critics described vampires in their reviews would be indicative of their feelings about the movies. In order to do so, I
# created co-occurrence networks to examine the 30 most frequent words that were 5 words or fewer away from the words "vampire" or "vampires."

# tokenize corpora; remove punctuation, numbers, and stopwords
fresh_tokens <- tokens(fresh_corpus, remove_punct = T, remove_numbers = T) |>
  tokens_select(pattern = stopwords(), select = "remove") |> tokens_tolower()
rotten_tokens <- tokens(rotten_corpus, remove_punct = T, remove_numbers = T) |>
  tokens_select(pattern = stopwords(), select = "remove") |> tokens_tolower()

# keep words surrounding "vampire" or "vampires" in reviews
fresh_near_vamp <- tokens_select(fresh_tokens, pattern = "vampire",
                                 window = 5, selection = "keep")
fresh_near_vamps <- tokens_select(fresh_tokens, pattern = "vampires",
                                  window = 5, selection = "keep")
rotten_near_vamp <- tokens_select(rotten_tokens, pattern = "vampire",
                                  window = 5, selection = "keep")
rotten_near_vamps <- tokens_select(rotten_tokens, pattern = "vampires",
                                   window = 5, selection = "keep")

# create feature co-occurrence matrices
fresh_fcm1 <- fcm(fresh_near_vamp, context = "window")
fresh_fcm2 <- fcm(fresh_near_vamps, context = "window")
rotten_fcm1 <- fcm(rotten_near_vamp, context = "window")
rotten_fcm2 <- fcm(rotten_near_vamps, context = "window")

# only keep 30 most frequent co-occurring words
top_fresh_feats1 <- names(sort(colSums(fresh_fcm1), decreasing = TRUE)[1:30]) 
vamp_fresh_fcm <- fcm_select(fresh_fcm1, pattern = c("vampire", top_fresh_feats1))
top_fresh_feats2 <- names(sort(colSums(fresh_fcm2), decreasing = TRUE)[1:30]) 
vamps_fresh_fcm <- fcm_select(fresh_fcm2, pattern = c("vampires", top_fresh_feats2))
top_rot_feats1 <- names(sort(colSums(rotten_fcm1), decreasing = TRUE)[1:30]) 
vamp_rot_fcm <- fcm_select(rotten_fcm1, pattern = c("vampire", top_rot_feats1))
top_rot_feats2 <- names(sort(colSums(rotten_fcm2), decreasing = TRUE)[1:30]) 
vamps_rot_fcm <- fcm_select(rotten_fcm2, pattern = c("vampires", top_rot_feats2))

set.seed(12)
textplot_network(vamp_fresh_fcm)
textplot_network(vamps_fresh_fcm)

# The co-occurrence networks of "vampire" and "vampires" for Fresh reviews include some positive words such as "original," "love," "best," and "new." There are also 
# some words like "Iranian," "Swedish," "Abraham," "Lincoln," "Twilight," and "Jarmusch" which are specific to certain movies.

set.seed(12)
textplot_network(vamp_rot_fcm)
textplot_network(vamps_rot_fcm)

# The co-occurrence networks of "vampire" and "vampires" for Rotten reviews include some negative words like "bad," "dull," and "suck," but they also feature some 
# positive words like "good," "better," "fun," and "right." Similarly to the co-occurrence networks for the positive reviews, there are some movie-specific words such
# as "Abraham," "Lincoln," "Academy," "Twilight," and "Underworld."

### Supervised Learning
# With a basic understanding of some of the most common words used in positive and negative reviews of vampire movies, I proceeded to apply supervised learning
# techniques to the data. My goal was to see how well these algorithms could predict the sentiment of the reviews and which words in the reviews were most strongly used
# to signal the sentiment.

# make one big corpus
vamp_reviews$review_ID <- seq(1:nrow(vamp_reviews))
vamp_corpus <- corpus(vamp_reviews, docid_field = "review_ID",
                      text_field = "review_content")
docvars(vamp_corpus, "review_ID") <- vamp_reviews$review_ID
ndoc(vamp_corpus)

# set seed, then split into training and testing sets
set.seed(12)
N <- ndoc(vamp_corpus)
trainIndex <- sample(1:N, .8*N)
testIndex <- c(1:N)[-trainIndex]

# create dfms for different subsets
dfmTrain <- corpus_subset(vamp_corpus, review_ID %in% trainIndex) |>
  tokens() |> dfm() |> dfm_trim(min_docfreq = 20)
dim(dfmTrain) # 80% of all documents
dfmTest <- corpus_subset(vamp_corpus, review_ID %in% testIndex) |>
  tokens() |> dfm()
dim(dfmTest) # 20% of all documents

#### Naive Bayes Model
# run Naive Bayes model on the training data
nb_model <- textmodel_nb(dfmTrain, docvars(dfmTrain, "review_freshness"),
                         distribution = "Bernoulli")
summary(nb_model)

# evaluate performance of model on test data
dfmTestMatched <- dfm_match(dfmTest, features = featnames(dfmTrain))
actual <- docvars(dfmTestMatched, "review_freshness")
predicted <- predict(nb_model, newdata = dfmTestMatched)
confusion <- table(predicted, actual)
confusionMatrix(confusion)

# From these initial results, we can see that the Naive Bayes model performed fairly well, with an accuracy of 72%. This is significantly better than classifying all 
# reviews are positive, indicated by the 21% increase from the No Information Rate of 51%.

# compare probabilities of features belonging to positive or negative reviews
nb_feature_probs <- as.data.frame(t(nb_model$param))
nb_feature_probs$Difference <- nb_feature_probs$Fresh - nb_feature_probs$Rotten

# examine words that are more positive than negative
head(arrange(nb_feature_probs, desc(Difference)))

# examine words that are more negative than positive
head(arrange(nb_feature_probs, Difference))

# Many of the words that are shown to be the "most positive" or the "most negative," based on the difference in the probability they belong to positive or negative
# reviews, are stopwords or punctuation. In addition to traditional stopwords like "and," "a," "in," "to," "so," and "or," "vampire" can also be considered a
# stopword in the context of this corpus. It is interesting that "horror" tends to be used more often in positive reviews, though, while it makes sense for "not" and
# "bad" to be used more often in negative reviews.
# Given these initial results, I wanted to re-run the Naive Bayes model without including stopwords or punctuation in the document-feature matrices, to see how that
# would affect the results.

# create dfms without stopwords or punctuation
dfmTrain <- corpus_subset(vamp_corpus, review_ID %in% trainIndex) |>
  tokens(remove_punct=T) |> tokens_select(c("vampire", "vampires"), selection="remove") |>
  dfm() |> dfm_remove(stopwords('english')) |> dfm_trim(min_docfreq = 20)
dim(dfmTrain) # 80% of all documents
dfmTest <- corpus_subset(vamp_corpus, review_ID %in% testIndex) |>
  tokens(remove_punct=T) |> tokens_select(c("vampire", "vampires"), selection="remove") |>
  dfm() |> dfm_remove(stopwords('english'))
dim(dfmTest) # 20% of all documents

# run NB model with no stop words
nb_model_nsw <- textmodel_nb(dfmTrain, docvars(dfmTrain, "review_freshness"),
                          distribution = "Bernoulli")
summary(nb_model_nsw)

# evaluate performance of new model on test data
dfmTestMatched <- dfm_match(dfmTest, features = featnames(dfmTrain))
actual <- docvars(dfmTestMatched, "review_freshness")
predicted <- predict(nb_model_nsw, newdata = dfmTestMatched)
confusion <- table(predicted, actual)
confusionMatrix(confusion)

# compare probabilities of features
nb_feature_probs <- as.data.frame(t(nb_model_nsw$param))
nb_feature_probs$Difference <- nb_feature_probs$Fresh - nb_feature_probs$Rotten

# examine words that are more positive than negative
head(arrange(nb_feature_probs, desc(Difference)))

# examine words that are more negative than positive
head(arrange(nb_feature_probs, Difference))

# Removing the stopwords and punctuation from the DFMs resulted in a slightly lower classification accuracy of around 70%. This time, the most positive and most 
# negative words make more sense, however. Words like "fun" and "best" are more likely to appear in positive reviews, while words like "bad," "worst," and "mess" 
# are more likely to appear in negative reviews. It's also interesting to note that positive reviews are more likely to refer to their objects of critique as "films,"
# while negative reviews refer to them as "movies," which doesn't imply as high of an artistic value as the word "film" does.

#### Support Vector Machines
# use same DFM used for NB model (trimmed, doesn't contain punctuation or stopwords)
dim(dfmTrain)

# run SVM model
svm_model <- textmodel_svm(dfmTrain, docvars(dfmTrain, "review_freshness"))

# evaluate performance of model on test data
dfmTestMatched <- dfm_match(dfmTest, features = featnames(dfmTrain))
actual <- docvars(dfmTestMatched, "review_freshness")
predicted <- predict(svm_model, newdata = dfmTestMatched)
confusion <- table(predicted, actual)
confusionMatrix(confusion)

# With an accuracy around 67%, the SVM model performed slightly worse than the Naive Bayes model, but not by much.

# examine most positive and most negative features
svmCoefs <- as.data.frame(t(coefficients(svm_model)))
svmCoefs <- svmCoefs |> arrange(V1)
tail(svmCoefs, 20)
head(svmCoefs, 20)

# The top 20 most positive and top 20 most negative features from the SVM model make intuitive sense. Some of the most positive features include "delight," 
# "terrific," "brilliant," "enjoyable," "succeeds," and "greatest." Some of the most negative features include "bland," "tedious," "neither," "dull," "poor,"
# and "stupid." There are also certain words, like "Amirpour," "Jarmusch," and "Nosferatu," which are included in the top 20 positive features because of the 
# creators or characters of specific highly-rated films and are not applicable to vampire movies as a whole.

#### Random Forests

# convert DFMs to matrices to run random forest model
dfmTrainMatrix <- convert(dfmTrain, to = "matrix")
dim(dfmTrainMatrix)
dfmTestMatchedMatrix <- convert(dfmTestMatched, to = "matrix")
dim(dfmTestMatchedMatrix)

# build RF model using 25 features per tree and 100 trees
rf_model <- randomForest(dfmTrainMatrix,
                         y = as.factor(docvars(dfmTrain)$review_freshness),
                         xtest = dfmTestMatchedMatrix,
                         ytest = as.factor(docvars(dfmTestMatched)$review_freshness),
                         importance = TRUE, mtry = 25, ntree = 100)

# evaluate model on test data
actual <- as.factor(docvars(dfmTestMatched)$review_freshness)
predicted <- rf_model$test[['predicted']]
confusion <- table(predicted, actual)
confusionMatrix(confusion)

# With an accuracy of 68%, the Random Forest model performed equally as well as the SVM model and just slightly worse than the Naive Bayes model.

# compare feature importances
varImpPlot(rf_model)

# According to the variable importance plot, some of the most important features that we can assume to be negative are: "mess," "worst," "tedious," "dull,"
# and "boring." Some of the most important features that we can assume to be positive are: "fun," "enjoyable," "best," "pleasure," and "entertaining."

### Conclusions
# Out of the three supervised learning models that I applied to my data, the Naive Bayes model performed the best, with a classification accuracy of 70%. 
# The Random Forest and Support Vector Machine models were close behind, with accuracies of 68% and 67%, respectively. Since all of these results were obtained 
# using only one training and testing set of data, I need to rerun all of these models on different random subsets of the data to see whether I get similar 
# results. I could also develop an ensemble model, giving equal weight to the Naive Bayes, Random Forest, and SVM models, since they all performed relatively
# equally, to see if those predictions would be at all improved over those from the individual models.
# Throughout my analysis, I have repeatedly come across the issue of the high weight that is given to words about specific vampire movies that are almost exclusively
# positively or negatively reviewed. For example, words that are associated with *A Girl Walks Home Alone at Night*, directed by Ana Lily Amirpour (including "Iranian"
# and "Amirpour"), *Only Lovers Left Alive*, directed by Jim Jarmusch (including "Jarmusch" and "Jarmusch's"), and *What We Do in the Shadows*, directed by Taika 
# Waititi and Jemaine Clement (including "shadows" and "mockumentary"), are repeatedly featured in the positive word analyses. Likewise, words that are associated 
# with the *Twilight*, *Underworld*, and *Blade* series are repeatedly featured in the negative word analyses. I considered removing references to specific movie 
# titles, characters, or creative personnel from my corpora, but that may have led to a loss of information in reviews that refer to these other movies in a positive
# or negative comparative way.
# Overall, I saw evidence to support both of my initial hypotheses. Fresh reviews of vampire movies frequently included words such as "good," "great," "best," "fun,"
# "original," "love," "entertaining," "classic," and "pleasure." Rotten reviews of vampire movies frequently included words such as "bad," "dull," "poor," "boring," 
# "worst," "nothing," "mess," "tedious," and "flat." These results align with what I expected to see from my first hypothesis. Additionally, the word "horror" was 
# repeatedly associated more with Fresh reviews than Rotten reviews, which aligns with my second hypothesis.
# In the future, I would like to apply my supervised learning models to reviews of newer vampire movies on Rotten Tomatoes that were published after 2020 and thus 
# weren't included in my dataset. I would be curious to see how well the models would perform on this new data to determine whether the results I've observed so far 
# could be considered generalizable or not.
