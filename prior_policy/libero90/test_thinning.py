import unittest
from thinning import select_rows


class ThinningTests(unittest.TestCase):
    def test_episode_isolation_and_last(self):
        rows = [('a',i,i-1) for i in range(1,21)] + [('b',i,i-1) for i in range(1,5)]
        chosen = select_rows(rows,8)
        self.assertEqual([r[1] for r in chosen if r[0]=='a'], [1,9,17,20])
        self.assertEqual([r[1] for r in chosen if r[0]=='b'], [1,4])
        self.assertTrue(all(row in rows for row in chosen))

    def test_stride_one_preserves_data(self):
        rows=[('a',i,i) for i in range(11)]
        self.assertEqual(select_rows(rows,1),rows)

    def test_terminal_not_duplicated(self):
        self.assertEqual(len(select_rows([('a',i,i) for i in range(9)],8)),2)

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError): select_rows([('a',1,0),('a',1,0)])

    def test_empty(self):
        self.assertEqual(select_rows([]),[])


if __name__=='__main__': unittest.main()
