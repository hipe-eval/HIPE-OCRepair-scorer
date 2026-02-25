import json
from spellchecker import SpellChecker
from hipe_ocrepair_scorer.ocrepair_eval import Evaluation


class SpellCheckCorrect():

    def __init__(self):

        self.name = "spellcheck"
        self.spellcheckers = {}
        self.spellcheckers["en"] = SpellChecker()  # the default is English (language='en')
        self.spellcheckers["fr"] = SpellChecker(language='es')  # use the Spanish Dictionary
        self.spellcheckers["de"] = SpellChecker(language='de')

    def correct_text(self, sentence, lang):
        spell = self.spellcheckers[lang]
        words = spell.split_words(sentence)
        misspelled = {w for w in spell.unknown(words) if len(w) < 8}
        new = []
        for word in words:
            if word not in misspelled:
                new.append(word)
                continue
            correction = spell.correction(word)
            if not correction:
                new.append(word)
                continue
            new.append(correction)
        return " ".join(new)


class DummyCorrectReproduce():

    def __init__(self):
        self.name = "same"

    @staticmethod
    def correct_text(sentence, lang):
        return sentence


class DummyCorrectRandom():

    def __init__(self):
        from random import seed, shuffle
        self.shuffle = shuffle
        self.name = "random"
        seed(42)

    def correct_text(self, sentence, lang):
        sentence = sentence.split()
        self.shuffle(sentence)
        sentence = " ".join(sentence)
        return sentence


if __name__ == "__main__":

    # We load data
    dataroot = "data/datasets/converted/v0.2/"
    datafiles = [dataroot + "impresso-nzz/de/ocr-post-correction-v0.2-impresso-nzz-test-de.jsonl",
                 dataroot + "icdar-2019/de/ocr-post-correction-v0.2-icdar-2019-test-de.jsonl",
                 dataroot + "overproof/en/ocr-post-correction-v0.2-overproof-test-en.jsonl"]

    # We get all jsonlines from all data sets.
    # Each line contains one post-correction challenge.
    jsonlines = []
    for filepath in datafiles:
        with open(filepath, "r") as f:
            for line in f:
                dic = json.loads(line)
                jsonlines.append(dic)

    # We instantiate a system this system
    system = DummyCorrectReproduce()

    # We use the system to predict a refined output
    # for each initial OCR hypothesis
    for jl in jsonlines:
        # retrieve challenge
        hyp = jl["ocr_hypothesis"]["transcription_unit"]
        language = jl["document_metadata"]["language"]

        # we apply the system
        post_correct = system.correct_text(hyp, language)

        # add output as "transcription unit"
        jl["ocr_postcorrection_output"]["transcription_unit"] = post_correct

        # add meta information about the applied system
        jl["ocr_postcorrection_output"]["ocr_postcorrection_system"] = system.name

    # We finally evaluate the predictions

    evaluator = Evaluation(jsonlines)
    mainresult = evaluator.score_over_datasets()
    latexstring = evaluator.scores2latex(mainresult, system.name)
    print(mainresult)
    print(latexstring)
