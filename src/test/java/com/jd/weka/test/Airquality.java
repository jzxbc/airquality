package com.jd.weka.test;

import com.demo.WekaUtil;
import org.junit.Test;
import weka.attributeSelection.*;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.SimpleLinearRegression;
import weka.classifiers.trees.*;
import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Airquality {

    private Instances getInstances(String location) throws Exception {
        Instances data = ConverterUtils.DataSource.read(location);
        // 去掉头信息
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }
    private void saveArff(Instances instances, String targetLocation) throws Exception {
        // 保存arff格式
        ArffSaver saver = new ArffSaver();
        saver.setInstances(instances);
        saver.setFile(new File(targetLocation));
        saver.writeBatch();
    }

    @Test
    public void loadData() throws Exception {
        String location = "data/airquality.csv";
        // 1 加载数据集，转换arff格式，并处理格式等问题
        FileReader fileReader = new FileReader(location);
        BufferedReader bufferedReader = new BufferedReader(fileReader);

        String attributeStr = bufferedReader.readLine();
        List<String> dataList = new ArrayList<>();

        String str;
        while ((str = bufferedReader.readLine()) != null) {
            if (!str.isEmpty()) {
                dataList.add(str);
            }
        }
        String[] attributeArray = attributeStr.split(",");

        ArrayList<Attribute> attributeList = new ArrayList<>(attributeArray.length);
        for (int i = 0; i < attributeArray.length; i++) {
            if (i == 0 || i == 1 || i == 11) {
                // 字符串类型
                attributeList.add(new Attribute(attributeArray[i], (ArrayList<String>) null));
            } else {
                // 数字类型
                attributeList.add(new Attribute(attributeArray[i]));
            }
        }
        Instances instances = new Instances("airquality", attributeList, 0);
        // 赋值
        for (int i = 0; i <= dataList.size() - 1; i++) {
            String[] data = dataList.get(i).split(",");
            double[] values = new double[attributeArray.length];
            for (int j = 0; j < attributeList.size(); j++) {
                if (j == 0 || j == 1 || j == 11) {
                    values[j] = instances.attribute(j).addStringValue(data[j]);
                } else {
                    values[j] = new Double(data[j]);
                }
            }
            instances.add(new DenseInstance(1.0, values));
        }
        System.out.println("加载数据集------------------------------------------------");
        System.out.println(instances);
        this.saveArff(instances, "data/airquality1.arff");
    }

    // 删除无用属性
    @Test
    public void remove() throws Exception {
        Instances instances = this.getInstances("data/airquality1.arff");
        Remove remove = new Remove();
        remove.setAttributeIndicesArray(new int[]{0, 1, 2, 11});
        remove.setInputFormat(instances);
        instances = Filter.useFilter(instances, remove);
        System.out.println("删除属性------------------------------------------------");
        System.out.println(instances);
        this.saveArff(instances, "data/airquality2.arff");
    }

    // 缺失值处理
    @Test
    public void replaceMissingValues() throws Exception {
        Instances instances = this.getInstances("data/airquality2.arff");
        ReplaceMissingValues values = new ReplaceMissingValues();
        values.setInputFormat(instances);
        instances = Filter.useFilter(instances, values);
        System.out.println("缺失值处理------------------------------------------------");
        System.out.println(instances);
        this.saveArff(instances, "data/airquality3.arff");
    }

    // 标准化处理
    @Test
    public void standardize() throws Exception {
        Instances instances = WekaUtil.getInstances("data/airquality3.arff");
        Standardize values = new Standardize();
        instances.setClassIndex(1);
        values.setInputFormat(instances);
        instances = Filter.useFilter(instances, values);
        System.out.println("标准化处理------------------------------------------------");
        System.out.println(instances);
        this.saveArff(instances, "data/airquality4.arff");
    }

    // 特征选择
    @Test
    public void cfsSubsetEval() throws Exception {
        Instances instances = this.getInstances("data/airquality4.arff");
        ASEvaluation evaluator = new CfsSubsetEval();
        ASSearch search = new BestFirst();
        AttributeSelection attributeSelection = new AttributeSelection();
        attributeSelection.setEvaluator(evaluator);
        attributeSelection.setSearch(search);
        instances.setClassIndex(1);
        attributeSelection.SelectAttributes(instances);
        System.out.println("特征选择------------------------------------------------");
        System.out.println(attributeSelection.toResultsString());
    }

    // 分类
    @Test
    public void m5p() throws Exception {
        Instances instances = this.getInstances("data/airquality4.arff");
        M5P classifier = new M5P();
        // 对文本实例分类
        instances.setClassIndex(1);
        classifier.buildClassifier(instances);
        Evaluation evaluation = new Evaluation(instances);
        // 采用十交叉验证的方法
        evaluation.crossValidateModel(classifier, instances, 20, new Random(1));
        System.out.println("分类------------------------------------------------");
        System.out.println(evaluation.toSummaryString());
    }

    // 回归
    @Test
    public void simpleLinearRegression() throws Exception {
        Instances instances = this.getInstances("data/airquality4.arff");
        SimpleLinearRegression classifier = new SimpleLinearRegression();
        // 对文本实例分类
        instances.setClassIndex(0);
        classifier.buildClassifier(instances);
        Evaluation evaluation = new Evaluation(instances);
        // 采用十交叉验证的方法
        evaluation.crossValidateModel(classifier, instances, 20, new Random(1));
        System.out.println("回归------------------------------------------------");
        System.out.println(evaluation.toSummaryString());
    }

}
