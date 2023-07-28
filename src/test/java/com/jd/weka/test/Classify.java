package com.jd.weka.test;

import com.demo.WekaUtil;
import org.junit.Test;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomTree;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Random;

/**
 * @author 程楠
 * @date 2023/7/15
 */
public class Classify {

    // randomTree
    @Test
    public void randomTree() throws Exception {
        Instances data = WekaUtil.getInstances("data/diabetes.arff");
        RandomTree classifier = new RandomTree();
        // 对文本实例分类
        classifier.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        // 采用十交叉验证的方法
        eval.crossValidateModel(classifier, data, 10, new Random(1));

        System.out.println(eval.toSummaryString());
    }

    // j48
    @Test
    public void j48() throws Exception {
        Instances data = WekaUtil.getInstances("data/diabetes.arff");
        J48 classifier = new J48();
        // 对文本实例分类
        classifier.buildClassifier(data);

        Evaluation eval = new Evaluation(data);
        // 采用十交叉验证的方法
        eval.crossValidateModel(classifier, data, 10, new Random(1));

        System.out.println(eval.toSummaryString());
    }

}
