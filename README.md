# Smart Ad Classifier - NLP-Powered Craigslist Moderation

## Overview
This project applies Natural Language Processing (NLP) to automatically classify and flag miscategorized listings in Craigslist’s **Computers** category.

## It Includes
- Web scraping with **Selenium** & **BeautifulSoup** to collect and label ads.  
- **TF-IDF vectorization** (unigrams + bigrams) for feature extraction.  
- **Logistic Regression** model for accurate, interpretable classification.  
- Confidence-based flagging of borderline cases for manual review.  
- [Streamlit app](YOUR_STREAMLIT_LINK) for interactive predictions.  

## Outcomes
- **Cleaner Search Results** – Separates computers from accessories and parts, improving buyer experience.  
- **Moderator Efficiency** – Flags low-confidence cases, reducing manual review workload.  
- **Scalable Approach** – Framework can be adapted to other categories (Jobs, Cars, Housing, etc.).  
- **Interpretable Model** – Keyword-based features aid transparency in decision-making.  

## License
This project is licensed under the MIT License.
