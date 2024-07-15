# This is a python script for performing feature extraction on email files.
# Ran succesfully on total 5844 email files present in 2 local directories for the project.

import numpy as np
import pandas as pd
import os
import re
import email
import chardet
from nltk import PorterStemmer


class EmailFeatureExtractor:
    def __init__(self):
        self.NOT_SPAM, self.SPAM = 0, 1
        self.stemmer = PorterStemmer()  # Global word stemmer object

        self.word_list = [
            "money",
            "income",
            "bankruptcy",
            "credit",
            "debt",
            "all",
            "full",
            "refund",
            "claims",
            "deal",
            "card",
            "click here",
            "deal",
            "exclusive",
            "apply online",
            "winner",
            "cash",
            "fast",
            "earn",
            "guaranteed",
            "bonus",
            "urgent",
            "send",
            "click",
            "be a member",
            "member",
            "investment",
            "profit",
            "access",
            "subscribe",
        ]

        self.char_list = ["!", "#", "$", ")", "["]

        self.word_freq_column_list = [f"word_freq_{word}" for word in self.word_list]
        self.char_freq_column_list = [f"char_freq_{char}" for char in self.char_list]

        self.column_names = (
            self.word_freq_column_list
            + self.char_freq_column_list
            + [
                "link_freq",
                "num_words",
                "length_of_text",
                "longest_word_length",
                "num_of_html_tags",
                "is_html",
                "target",
            ]
        )

        self.data = []  # This list stores all data rows

    def process_directory(self, directory_path, output_file="data_final.csv"):
        directories = {"ham": self.NOT_SPAM, "spam": self.SPAM}
        progress = 0

        for directory, label in directories.items():
            dir_path = os.path.join(directory_path, directory)

            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                try:
                    self.add_data(file_path, label)
                except UnicodeDecodeError as e:
                    print(f"UnicodeDecodeError occurred in file: {file_path}")
                    print(f"Error details: {str(e)}")
                except Exception as e:
                    print(f"Error occurred while processing file: {file_path}")
                    print(f"Error details: {str(e)}")
                progress += 1
                if progress % 500 == 0:
                    print(f"Number of files processed: {progress}")

        df = pd.DataFrame(self.data)  # Convert the list of data rows into a DataFrame object
        df.to_csv(output_file, index=False)  # Save DataFrame as a csv file

    def add_data(self, file_path, label):
        """
        Appends the features and the label of this particular training example (single email) to the global
        list 'data'.
        Args:
            file_path: str, full file path of email to be processed
            label: int64, binary value corresponding to whether current email is ham (0) or spam (1)
        Returns: None
        Side Effect: new data row (dict) is added to 'data'

        Note:
            'features': Dict[str, int64], feature vector of email
            'email_content': str, body of email
            'words_in_email': List[str], space seperated words in body of email
            'stemmed_words': List[str], list of all words in the email body stemmed down to their root word
        """
        features = {
            feature: 0 for feature in self.column_names
        }  # Initialise feature vector

        with open(file_path, "r") as f:
            raw_email = f.read()  # Gives contents of file in str datatype

        email_message, email_content = self.parse_email(raw_email)

        email_file_contents = email_content
        words_in_email = np.array(
            email_file_contents.split()
        )  # Get all space separated words

        stemmed_words = self.stem_words(words_in_email)

        for stemmed_word in stemmed_words:
            for word in self.word_list:
                if stemmed_word == self.stemmer.stem(word):
                    features[f"word_freq_{word}"] += 1

        for char in self.char_list:
            features[f"char_freq_{char}"] = np.char.count(email_file_contents, char)

        features["length_of_text"] = np.sum([len(word) for word in words_in_email])
        features["longest_word_length"] = (
            np.max(np.char.str_len(words_in_email)) if words_in_email.size > 0 else 0
        )
        features["num_words"] = words_in_email.size

        features["link_freq"] += self.num_links(email_file_contents)

        features["num_of_html_tags"] += self.num_html_tags(email_file_contents)

        features["is_html"] = 1 if features["num_of_html_tags"] != 0 else 0

        features["target"] = label

        self.data.append(features)

    def stem_words(self, words):
        """
        Performs word stemming (the process of reducing deviated words to their word stem).
        examples -
            banking -> bank
            randomness -> random
            preprocessing -> process
        Args:
            words: List[str], list of words occuring in email body
        Returns:
            stemmed_words: List[str], list of words occuring in email body, but reduced to their word stem

        Note:
            stemmer object (global): takes a word as input and returns its word stem
        """
        stemmed_words = []
        for word in words:
            stemmed_word = self.stemmer.stem(word)
            stemmed_words.append(stemmed_word)
        return stemmed_words

    def decode_payload(self, payload):
        """
        Decodes the email payload to a string, attempting to detect the correct encoding.
        We use the `chardet` library to detect the encoding of the email payload.
        If the encoding is detected successfully, it decodes the payload using that encoding,
        otherwise we default to UTF-8.

        Args:
            payload (bytes): The email body in bytes.

        Returns:
            str: The decoded email body as a string.

        Note:
            For simplicity's sake, any characters that cannot be decoded properly are ignored.
        """
        try:
            detection = chardet.detect(payload)
            encoding = detection.get("encoding", "utf-8")  # Try to detect encoding

            if encoding is None:
                encoding = "utf-8"  # Default to UTF-8 if detection fails

            return payload.decode(encoding, errors="ignore")  # Ignore decoding errors
        except (UnicodeDecodeError, LookupError):
            return payload.decode("utf-8", errors="ignore")

    def parse_email(self, raw_email):
        """
        Parses the raw email string and extracts the plain text content.

        Args:
            raw_email: str, raw email content.

        Returns:
            email_message: the email message object
            email_content: the plain text email content.

        Note:
            This function processes the raw email string to create an email message object.
            It checks if the email is multipart:
            - If it is multipart, it walks through the email parts to find the plain text part and decodes it.
            - If it is not multipart, it decodes the email payload directly.
            The function returns the email message object and the decoded plain text content.
        """
        email_message = email.message_from_string(raw_email)  # Created email message object
        email_content = ""

        if (email_message.is_multipart()):  # For multipart emails containing attachments etc
            for (part) in email_message.walk():  # Iterating through all the parts in the email
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    email_content = self.decode_payload(payload)
                    break
        else:  # For single part emails
            payload = email_message.get_payload(decode=True)
            email_content = self.decode_payload(payload)

        return email_message, email_content

    def num_html_tags(self, text):
        """
        Matches a regex to find number of occurences of html tags.
        Args:
            text: str, email plain text
        Returns:
            int64: number of html tags in email plain text
        """
        html_pattern = r"<[^>]+>"  # Regex for matching html pattern
        return len(re.findall(html_pattern, text))

    def num_links(self, text):
        """
        Matches a regex to find number of occurences of links or URLs.
        Args:
            text: str, email plain text
        Returns:
            int64: number of links or URLs in email plain text
        """
        link_pattern = (r"\b(?:https?://|www\.)\S+\b") # Regex for matching link/url pattern
        return len(re.findall(link_pattern, text))


if __name__ == "__main__":
    extractor = EmailFeatureExtractor()
    extractor.process_directory(os.getcwd())
