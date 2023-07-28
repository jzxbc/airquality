package com.jd.weka.test;

import com.demo.WekaUtil;
import org.junit.Test;
import weka.attributeSelection.*;
import weka.core.Instances;

/**
 * 特征选择
 *
 * @author 程楠
 * @date 2023/7/6
 */
public class FeatureSelector {


    // 单个属性评估器，排序没有提取
    @Test
    public void infoGainAttributeEval() throws Exception {
        Instances data = WekaUtil.getInstances("data/airquality4.arff");
        // 评估器评估每个属性的预测能力及其相互之间的冗余度，倾向于选择与类别属性相关度高，但相互之间相关度低的属性
        ASEvaluation evaluator = new InfoGainAttributeEval();
        // 搜索方法，实际上不是搜索属性子集的方法，而是对单个属性进行排名的方法。通过对单个属性评估对属性排序，
        // 只能用户单个属性评估器，不能用户属性子集评估器。
        ASSearch search = new Ranker();
        AttributeSelection eval = new AttributeSelection();
        eval.setEvaluator(evaluator);
        eval.setSearch(search);

        eval.SelectAttributes(data);
        System.out.println(eval.toResultsString());

    }

    // 估器评估每个属性的预测能力及其相互之间的冗余度，倾向于选择与类别属性相关度高，但相互之间相关度低的属性。
    // 选项迭代添加与类别属性相关度最高的属性，只要子集中不包含与当前属性相关度更高的属性。
    @Test
    public void cfsSubsetEval() throws Exception {
        Instances data = WekaUtil.getInstances("data/weather.nominal.arff");
        // 评估器评估每个属性的预测能力及其相互之间的冗余度，倾向于选择与类别属性相关度高，但相互之间相关度低的属性
        ASEvaluation evaluator = new CfsSubsetEval();
        // 搜索方法执行带回溯的贪婪爬山法。它可以从空属性集开始向前搜索，也可以从全集开始向后搜索，
        // 还可以从中间点（通过属性索引列表指定）开始双向搜索并考虑所有可能的单个属性的增删操作。
        ASSearch search = new BestFirst();
        AttributeSelection eval = new AttributeSelection();
        eval.setEvaluator(evaluator);
        eval.setSearch(search);

        eval.SelectAttributes(data);
        System.out.println(eval.toResultsString());

    }

    // 估器评估每个属性的预测能力及其相互之间的冗余度，倾向于选择与类别属性相关度高，但相互之间相关度低的属性。
    // 选项迭代添加与类别属性相关度最高的属性，只要子集中不包含与当前属性相关度更高的属性。
    @Test
    public void cfsSubsetEval2() throws Exception {
        Instances data = WekaUtil.getInstances("data/weather.nominal.arff");
        // 评估器评估每个属性的预测能力及其相互之间的冗余度，倾向于选择与类别属性相关度高，但相互之间相关度低的属性
        ASEvaluation evaluator = new CfsSubsetEval();
        // 搜索方法贪婪搜索属性的子集空间。像BestFirst 搜索方法一样，它可以向前和向后搜索。但是，它不进行回溯。
        // 只要添加或删除剩余的最佳属性导致评估指标降低，就立即终止。
        ASSearch search = new GreedyStepwise();
        AttributeSelection eval = new AttributeSelection();
        eval.setEvaluator(evaluator);
        eval.setSearch(search);

        eval.SelectAttributes(data);
        System.out.println(eval.toResultsString());

    }
}
