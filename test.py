import unittest
from InpGen import InputGenerator

class InpgenTest(unittest.TestCase):
    def test_create(self):
        gen = InputGenerator()
    def test_tuple(self):
        gen =(InputGenerator())
        self.assertEqual(len(next(gen)),2)

    def test_len(self):
        gen = (InputGenerator(batch_size=47))
        self.assertEqual(len(next(gen)[0]),47)
        self.assertEqual(len(next(gen)[1]),47)

    def test_size(self):
        gen =(InputGenerator(size=(153,40)))
        self.assertEqual((next(gen)[0][0].shape[1]),153)
        self.assertEqual((next(gen)[0][0].shape[0]),40)
        self.assertEqual((next(gen)[1][0].shape[1]),153)
        self.assertEqual((next(gen)[1][0].shape[0]),40)



if __name__ == '__main__':
        unittest.main()
