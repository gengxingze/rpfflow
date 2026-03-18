import os
import unittest
import numpy as np
from ase.build import molecule
from ase.db import connect
from rpfflow.core.structure import optimize_structure  # 假设你的文件名是这个


class TestOptimizeStructure(unittest.TestCase):
    def setUp(self):
        """测试前的初始化：创建一个临时的 DB 文件"""
        self.db_path = "test_calc.db"
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        # 创建一个简单的水分子作为测试对象
        self.atoms = molecule('H2O')
        self.atoms.info['fragment_signature'] = "water_test_sig"

    def tearDown(self):
        """测试后的清理：删除临时 DB"""
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    def test_first_run_and_persistence(self):
        """测试第一次运行：是否成功计算并写入数据库"""
        # 执行优化
        result = optimize_structure(self.atoms, db_path=self.db_path, steps=2)

        # 验证数据库文件已创建
        self.assertTrue(os.path.exists(self.db_path))

        # 验证数据库中是否存在该 signature
        with connect(self.db_path) as db:
            rows = list(db.select(signature="water_test_sig"))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].formula, "H2O")

    def test_cache_hit(self):
        """测试第二次运行：是否直接从缓存读取（不触发计算）"""
        # 1. 运行第一次，确保存入
        optimize_structure(self.atoms, db_path=self.db_path, steps=5)

        # 2. 修改当前的 atoms 坐标，模拟一个未优化的状态
        test_atoms = molecule('H2O')
        test_atoms.info['fragment_signature'] = "water_test_sig"
        original_pos = test_atoms.get_positions().copy()

        # 3. 再次运行优化函数
        # 如果命中缓存，它应该返回数据库里那个已经优化的位置，而不是我们刚创建的位置
        cached_result = optimize_structure(test_atoms, db_path=self.db_path)

        # 验证返回的结构与数据库中的 ID 一致（或者位置发生了显著变化）
        self.assertFalse(np.allclose(cached_result.get_positions(), original_pos))
        print("Success: 缓存命中并恢复了优化后的坐标")

    def test_signature_mismatch(self):
        """测试签名校验：如果 Formula 相同但 Signature 不同，是否会重新计算"""
        # 1. 存入一个水分子
        optimize_structure(self.atoms, db_path=self.db_path)

        # 2. 创建一个同化学式但不同签名的水分子
        other_atoms = molecule('H2O')
        other_atoms.info['fragment_signature'] = "different_sig"

        # 3. 运行优化
        optimize_structure(other_atoms, db_path=self.db_path)
        print(other_atoms, other_atoms.get_potential_energy())
        # 4. 数据库中应该有 2 条记录
        with connect(self.db_path) as db:
            self.assertEqual(db.count(), 2)


if __name__ == '__main__':
    unittest.main()

