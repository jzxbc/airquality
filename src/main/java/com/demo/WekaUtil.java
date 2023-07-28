package com.demo;

import weka.core.Instances;
import weka.core.converters.ConverterUtils;

/**
 * @author 程楠
 * @date 2023/7/15
 */
public class WekaUtil {

    /**
     * 加载数据集
     */
    public static Instances getInstances(String location) throws Exception {
        // ConverterUtils.DataSource source = new ConverterUtils.DataSource(location);
        // Instances data = source.getDataSet();

        Instances data = ConverterUtils.DataSource.read(location);
        // 去掉头信息
        if (data.classIndex() == -1) {
            data.setClassIndex(data.numAttributes() - 1);
        }
        return data;
    }
}
