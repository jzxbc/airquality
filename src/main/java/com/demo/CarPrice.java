package com.demo;

import weka.core.Attribute;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

public class CarPrice {

    public static void main(String[] args) {
        String location = "data/car/train1.csv";
        String targetLocation = "data/car/train1.arff";
        try {
            // 1. 加载数据集，转换arff格式，并处理格式等问题
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
//            System.out.println(attributeStr);
            String[] attributeArray = attributeStr.split(" ");

            ArrayList<Attribute> attributeList = new ArrayList<>(attributeArray.length);
            for (String attribute : attributeArray) {
                attributeList.add(new Attribute(attribute));
            }

            Instances instances = new Instances("car", attributeList, 0);
            instances.setClassIndex(instances.numAttributes() - 15);

            for (int i = 0; i <= dataList.size() - 1; i++) {
//                System.out.println(dataList.get(i));
                String[] data = dataList.get(i).split(" ");
                Instance instance = new DenseInstance(attributeArray.length);
                for (int j = 0; j < attributeList.size(); j++) {
                    if (!"-".equals(data[j]) && !"".equals(data[j])) {
//                        try {
                            instance.setValue(j, new Double(data[j]));
//                        } catch (Exception e) {
//                            e.printStackTrace();
//                            System.err.println("转换异常数据: " + data[j]);
//                            break;
//                        }
                    }
                }
                instances.add(instance);
            }
            System.out.println("加载数据集------------------------------------------------");
            System.out.println(instances);
            // 保存arff格式
            ArffSaver saver = new ArffSaver();
            saver.setInstances(instances);
            saver.setFile(new File(targetLocation));
            saver.writeBatch();

            // 2 数据预处理
            // 删除属性
            Remove remove = new Remove();
            remove.setAttributeIndices("1");
            remove.setInputFormat(instances);
            instances = Filter.useFilter(instances, remove);
            System.out.println("删除属性------------------------------------------------");
            System.out.println(instances);

            // 缺失值处理...
            ReplaceMissingValues values = new ReplaceMissingValues();
            values.setInputFormat(instances);
            instances = Filter.useFilter(instances, values);
            System.out.println("缺失值处理------------------------------------------------");
            System.out.println(instances);

            // 3. 训练模型

            // 4. 创建推荐实例

            // 5. 进行预测

            // 6. 输出推荐结果

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

}