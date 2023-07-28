package com.jd.weka.test;

import com.demo.WekaUtil;
import org.junit.Test;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.Remove;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;
import weka.filters.unsupervised.attribute.Standardize;

/**
 * 数据预处理
 *
 * @author 程楠
 * @date 2023/7/6
 */
public class DataPreprocess {

    // 缺失值处理
    @Test
    public void replaceMissingValues() throws Exception {
        Instances data = WekaUtil.getInstances("data/weather.nominal1.arff");
        ReplaceMissingValues values = new ReplaceMissingValues();
        values.setInputFormat(data);
        // 使用均值（数值型）和模式（我认为对于非数值则是数量最多的属性值填充）填充缺失值，默认跳过标签列（其中 ignoreClass 参数默认为 False）。
        values.setIgnoreClass(true);
        data = Filter.useFilter(data, values);
        System.out.println(data);
    }

    // 标准化
    @Test
    public void standardize() throws Exception {
        Instances data = WekaUtil.getInstances("data/cpu.arff");
        Standardize values = new Standardize();
        values.setInputFormat(data);
        data = Filter.useFilter(data, values);
        System.out.println(data);
    }

    // 规范化
    @Test
    public void normalize() throws Exception {
        Instances data = WekaUtil.getInstances("data/cpu.arff");
        Normalize values = new Normalize();
        values.setInputFormat(data);
        data = Filter.useFilter(data, values);
        System.out.println(data);
    }
}
