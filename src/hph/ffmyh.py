# -*- coding: utf-8 -*-
import os
import FeatureProcess
from sklearn.metrics import log_loss
#reload(FeatureProcess)
class FFM:
    def __init__(self, categorical, numerical, target, listype,
        reg_param = 0.00002,
        k = 4,
        iter_max = 15,
        learing_rate = 0.2,
        threads = 1,
        auto_stop = False,
        quiet = False,
        no_norm = False):


        self.trainer = "./ffm-train"
        self.predictor = "./ffm-predict"
        self.dir = "./ffm/"
        self.processor = FeatureProcess.FeatureProcess(   target=target, 

                            categorical=categorical, 

                            numerical=numerical,

                            listype=listype
                                )
        self.param = "-l %s -k %s -t %s -r %s -s %s" % (reg_param, k, iter_max, learing_rate, threads)

        if auto_stop:
            self.param += " --auto-stop"
        if quiet:
            self.param += " --quiet"
        if no_norm:
            self.param += " --no-norm"

        self.quiet = quiet
        self.categorical = categorical
        self.numerical = numerical
        self.target = target
        self.listype = listype

    
    def fit(self, train_df, validation_df=None):

        validation_param = ""
        is_cached, train_name = self.processor.readyCache(self.processor.toFFMData, train_df, path=self.dir, subfix=".txt")
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
            validation_param = "-p %s" % (self.dir + validation_name)

        self.model_name = train_name + ".model"
        self.train_name = train_name

        cmd = self.trainer + " %s %s %s %s" % (self.param, validation_param, self.dir + train_name, self.dir + self.model_name)
        if not self.quiet:
            print(cmd)
        output = os.popen(cmd).read()
        if not self.quiet:
            print(output)
        
        

    def predict_proba(self, test_df):
        return self.predict(test_df)
        
    def predict(self, test_df):

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







        