from tint.datasets import Mimic3
mimci3 = Mimic3()

mimci3.download(sqluser="postgres", split="train")
mimci3.download(sqluser="postgres", split="train")
