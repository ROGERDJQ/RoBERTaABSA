import argparse
import warnings
warnings.filterwarnings('ignore')

from dependency import _evaluation as dep_eval
from dependency.dep_parsing import decoding as dep_parsing
from dependency.dep_parsing import decoding_new as dep_parsing_new

from discourse.dis_parsing import decoding as dis_parsing
from discourse import evaluation as dis_eval
from discourse import distance_evaluation as dis_eval_per_distance
from constituency.con_parsing import decoding as con_parsing
from constituency.con_parsing import constituent_evaluation


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--matrix', default='./results/discourse/bert-dist-SciDTB-last.pkl')
    parser.add_argument('--probe', default='discourse', choices=['dependency', 'constituency', 'discourse', 'new_dep',
                                                                 'dep_save'])
    parser.add_argument('--decoder', default='eisner', choices=['eisner', 'cle', 'right_chain',
                                                                'top_down', 'mart', 'right_branching', 'left_branching'])
    parser.add_argument('--subword', default='avg', choices=['first', 'avg', 'max'])
    parser.add_argument('--root', default='gold', help='use gold root as init')
    parser.add_argument('--save_path', default='save_dep.txt', type=str, help="Where to save dep")

    args = parser.parse_args()
    print(args)

    if args.probe == 'dependency':
        trees, results, deprels = dep_parsing(args)
        dep_eval(trees, results)
    elif args.probe == 'new_dep':
        trees, results = dep_parsing_new(args)
        dep_eval(trees, results)
    elif args.probe == 'dep_save':
        trees, results = dep_parsing_new(args)
        # trees: [[(w_i, head_i), (w_i, head_i), ...], []]
        # results: [[line, 0, 0], [line, 0, 0]]
        with open(args.save_path, 'w', encoding='utf-8') as f:
            for ((line, _, _), tree) in zip(results, trees):
                sentence = [x.form for x in line][1:]  # [去掉root]
                tree = [head for (w_i, head) in tree[1:]]  # 去掉cls
                assert len(sentence)==len(tree)
                f.write(' '.join(sentence) + '\t' + ' '.join(map(str, tree)) + '\n')
    elif args.probe == 'dep_dist':  # 生成对应的文件
        pass
    elif args.probe == 'constituency':
        trees, results = con_parsing(args)
        constituent_evaluation(trees, results)
    elif args.probe == 'discourse':
        trees, gold_trees, deprels = dis_parsing(args)
        dis_eval(trees, gold_trees)
        dis_eval_per_distance(trees, gold_trees)
