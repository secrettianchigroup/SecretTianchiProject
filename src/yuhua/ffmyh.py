# -*- coding: utf-8 -*-
import os
import FeatureProcess
from sklearn.metrics import log_loss
import subprocess
import time
from itertools import combinations
import collections
# reload(FeatureProcess)
from prettytable import PrettyTable

# x = PrettyTable(["City name", "Area", "Population", "Annual Rainfall"])

# x.align["City name"] = "l" # Left align city names

# x.padding_width = 1 # One space between column edges and contents (default)

# x.add_row(["Adelaide",1295, 1158259, 600.5])

# x.add_row(["Brisbane",5905, 1857594, 1146.4])

# x.add_row(["Darwin", 112, 120900, 1714.7])

# x.add_row(["Hobart", 1357, 205556, 619.5])

# x.add_row(["Sydney", 2058, 4336374, 1214.8])

# x.add_row(["Melbourne", 1566, 3806092, 646.9])



class FFM:
    def __init__(self, processor,
        reg_param = 0.00002,
        k = 4,
        iter_max = 15,
        learing_rate = 0.2,
        threads = 1,
        auto_stop = False,
        quiet = False):


        self.trainer = "./ffm-train"
        self.predictor = "./ffm-predict"
        self.dir = "./ffm/"
        self.processor = processor
        self.param = [self.trainer]
        self.param.append("-l")
        self.param.append(reg_param)

        self.param.append("-k")
        self.param.append(k)

        self.param.append("-t")
        self.param.append(iter_max)

        self.param.append("-r")
        self.param.append(learing_rate)

        self.param.append("-s")
        self.param.append(threads)

        if auto_stop:
            self.param.append("--auto-stop")
        if quiet:
            self.param.append("--quiet")
        #nerver ever use the fucking ffm-norm
        # self.param.append("--no-norm")

        self.quiet = quiet
        self.categorical = processor.categorical
        self.numerical = processor.numerical
        self.target = processor.target
        self.listype = processor.listype

        self.tr = self.categorical+self.numerical+[self.target]+self.listype
        self.ts = self.categorical+self.numerical+self.listype

    def fit_schema(self):
        #获取训练集的所有
        self.ff_index, self.field_index, self.feature_index = self.processor.copyFFFIndexs()

        #计算反向index
        self.rev_feature_index = {}
        for k,v in self.feature_index.items():
            self.rev_feature_index[v] = k

        self.rev_field_index = {}
        for k,v in self.field_index.items():
            self.rev_field_index[v] = k

    
    def fit(self, train_df, validation_df=None):
        train_df = train_df[self.tr]
        if type(validation_df) != type(None):
            validation_df = validation_df[self.tr]
        validation_param = ""
        is_cached, train_name = self.processor.readyCache(self.processor.toFFMData, train_df, path=self.dir, subfix=".txt")

        #获取到训练集的schema
        self.fit_schema()


        if not is_cached :
            self.processor.toFFMData(train_df, self.dir + train_name)
        if not self.quiet:
            print(train_name)
        if type(validation_df) != type(None):
            is_cached, validation_name = self.processor.readyCache(self.processor.toFFMData, validation_df, path=self.dir, subfix=".txt")
            if not is_cached :
                self.processor.toFFMData(validation_df, self.dir + validation_name)

            if not self.quiet:
                print(validation_name)
            self.validation_name = validation_name

            self.param.append("-p")
            self.param.append(self.dir + validation_name)

        self.model_name = train_name + ".model"
        self.train_name = train_name


        self.param.append(self.dir + train_name)
        self.param.append(self.dir + self.model_name)


        for idx, v in enumerate(self.param):
            self.param[idx] = str(v)
        if not self.quiet:
            print(' '.join(self.param))

        popen = subprocess.Popen(self.param, stdout = subprocess.PIPE)

        while True:
            ss = popen.stdout.readline()
            if not ss:
                break

            if not self.quiet:
                print(ss.replace("\n", ""))

        while popen.poll() == None:
            time.sleep(0.1)
        
        

    def predict_proba(self, test_df):
        return self.predict(test_df)

    def parse_model(self, index_path, model_path, feaures_message=False):

        fp = open(index_path, "wb+")
        
        corss_features = list(combinations(self.feature_index.values(), 2))
        
        for fea in corss_features:
            j1 = fea[0]
            j2 = fea[1]

            f1 = self.ff_index[j1]
            f2 = self.ff_index[j2]

            if j1 == j2 or f1 == f2:
                continue


            new_list = sorted([f1, f2])
            if new_list[0] == f2:
                tmp = f1
                f1 = f2
                f2 = tmp

                tmp = j1
                j1 = j2
                j2 = tmp

            fp.write("%s:%s:%s:%s\n" % (f1,j1,f2,j2))

        fp.close()

        cmd = "./ffm-parser %s %s" % (index_path, model_path)

        output = os.popen(cmd).read()

        pos_sum_weight_sfd = collections.defaultdict(float)
        neg_sum_weight_sfd = collections.defaultdict(float)

        pos_sum_weight_fd = collections.defaultdict(float)
        neg_sum_weight_fd = collections.defaultdict(float)

        pos_sum_weight_fea = collections.defaultdict(float)
        neg_sum_weight_fea = collections.defaultdict(float)

        for line in output.split("\n"):
            line = line.strip()
            if not line:
                continue

            line = line.split(" ")
            weight = float(line[1])
            line = line[0]
            line = line.split("*")
            f1, j1 = line[0].split(":")
            f2, j2 = line[1].split(":")


            j1 = int(j1)
            j2 = int(j2)
            f1 = int(f1)
            f2 = int(f2)

            j1 = self.rev_feature_index[j1]
            j2 = self.rev_feature_index[j2]

            f1 = self.rev_field_index[f1]
            f2 = self.rev_field_index[f2]

            fd = "%s + %s" % (f1,f2)
            fea = "%s:%s + %s:%s" % (f1, j1, f2, j2)


            if weight >= 0:
                pos_sum_weight_sfd[f1] += weight
                pos_sum_weight_sfd[f2] += weight

                pos_sum_weight_fd[fd] += weight
            else:
                neg_sum_weight_sfd[f1] += weight
                neg_sum_weight_sfd[f2] += weight

                neg_sum_weight_fd[fd] += weight

            if weight >= 0:
                pos_sum_weight_fea[fea] += weight
            else:
                neg_sum_weight_fea[fea] += weight

        def __print_dict(d, typ):
            dd = sum(d.values())
            z = sorted(d.items(), key=lambda x:abs(x[1]), reverse=True)

            x = PrettyTable([typ, "value", "proportion"])

            x.align[typ] = "l" # Left align city names

            x.padding_width = 5 # One space between column edges and contents (default)

            for k,v in z:
                s = "%.5f" % ((v/dd) * 100)
                s += "%"
                x.add_row([k,v,s])

            print(x)

        # print "\n\n"
        # for k,v in self.field_index.items():
        #     print k,v
        # for k,v in self.feature_index.items():
        #     print k,v 
        # print "\n\n"


        __print_dict(pos_sum_weight_sfd, "正向field")

        print "\n-----------------------分割-----------------------\n"

        __print_dict(neg_sum_weight_sfd, "负向field")

        print "\n-----------------------分割-----------------------\n"

        __print_dict(pos_sum_weight_fd, "正向组合")

        print "\n-----------------------分割-----------------------\n"

        __print_dict(neg_sum_weight_fd, "负向组合")

        if feaures_message:
            print "\n-----------------------分割-----------------------\n"

            __print_dict(pos_sum_weight_fea, "正向交叉特征")

            print "\n-----------------------分割-----------------------\n"

            __print_dict(neg_sum_weight_fea, "负向交叉特征")

    def predict(self, test_df):
        test_df = test_df[self.ts]

        is_cached, test_name = self.processor.readyCache(self.processor.toFFMData, test_df, path=self.dir, subfix=".txt")
        if not is_cached :
            self.processor.toFFMData(test_df, self.dir + test_name)

        self.output_name = self.model_name + "*" + test_name + ".output"

        cmd = self.predictor + " %s %s %s " % (self.dir + test_name, self.dir + self.model_name, self.dir + self.output_name)
        output = os.popen(cmd).read()
        if not self.quiet:
            print(cmd)
            print(output)

        fp = open(self.dir + self.output_name, "rb+")
        y_predict = []
        while True:
            line = fp.readline()
            if not line:
                break
            y_predict.append(float(line.split(" ")[0]))
        return y_predict







        