package com.digquant;

import com.google.common.collect.ConcurrentHashMultiset;
import com.google.common.collect.Multiset;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.mahout.clustering.Cluster;
import org.apache.mahout.clustering.classify.ClusterClassifier;
import org.apache.mahout.clustering.iterator.KMeansClusteringPolicy;
import org.apache.mahout.clustering.kmeans.Kluster;
import org.apache.mahout.common.distance.CosineDistanceMeasure;
import org.apache.mahout.common.distance.DistanceMeasure;
import org.apache.mahout.math.NamedVector;
import org.apache.mahout.math.RandomAccessSparseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.vectorizer.encoders.FeatureVectorEncoder;
import org.apache.mahout.vectorizer.encoders.StaticWordValueEncoder;
import org.wltea.analyzer.lucene.IKAnalyzer;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class ClusterBook {

    public static List<NamedVector> loadData() throws IOException {
        HashMap<String, String> bookMap = new HashMap<>();
        {


            bookMap.put("Java程序性能优化实战", "本书以Java性能调优为主线，系统地阐述了与Java性能优化相关的知识与技巧。本书共6章，" +
                    "先后从软件设计、软件编码、JVM调优及程序故障排除等方面介绍针对Java程序的优化方法。" +
                    "第1章介绍性能的基本概念、木桶原理与Amdahl定律、系统调优的过程和注意事项；" +
                    "第2章从设计层面介绍与性能相关的设计模式及常用优化组件；第3章从代码层面介绍如何编写高性能的Java程序；" +
                    "第4章介绍并进行开发和如何通过多线程提高系统性能；第5章立足于JVM虚拟机层面，" +
                    "介绍如何通过设置合理的JVM参数提升Java程序的性能；第6章为工具篇，介绍了获取和监控程序或系统性能指标的各种工具，" +
                    "包括相关的故障排查工具。本书适合所有Java程序员、软件设计师、架构师及软件开发爱好者，" +
                    "对于有一定经验的Java工程师，本书更能帮助他突破技术瓶颈，深入Java内核开发");

            bookMap.put("Java编程思想(第4版)", "《Java编程思想（第4版）》赢得了全球程序员的广泛赞誉，即使是最晦涩的概念，" +
                    "在Bruce Eckel的文字亲和力和小而直接的编程示例面前也会化解于无形。从Java的基础语法到最高级特性" +
                    "（深入的面向对象概念、多线程、自动项目构建、单元测试和调试等），《Java编程思想（第4版）》都能逐步指导你轻松掌握。\n" +
                    "从java编程思想这本书获得的各项大奖以及来自世界各地的读者评论中，不难看出这是一本经典之作。本书的作者拥有多年教学经验，" +
                    "对C、C++以及Java语言都有独到、深入的见解，以通俗易懂及小而直接的示例解释了一个个晦涩抽象的概念。《Java编程思想（第4版）》" +
                    "共22章，包括操作符、控制执行流程、访问权限控制、复用类、多态、接口、通过异常处理错误、字符串、泛型、数组、容器深入研究、" +
                    "Java I/O系统、枚举类型、并发以及图形化用户界面等内容。这些丰富的内容，包含了Java语言基础语法以及高级特性，" +
                    "适合各个层次的Java程序员阅读，同时也是高等院校讲授面向对象程序设计语言以及Java语言的绝佳教材和参考书。");

            bookMap.put("Java并发编程实战", "《Java并发编程实战》深入浅出地介绍了Java线程和并发，" +
                    "是一本完美的Java并发参考手册。书中从并发性和线程安全性的基本概念出发，" +
                    "介绍了如何使用类库提供的基本并发构建块，用于避免并发危险、构造线程安全的类及验证线程安全的规则，" +
                    "如何将小的线程安全类组合成更大的线程安全类，如何利用线程来提高并发应用程序的吞吐量，" +
                    "如何识别可并行执行的任务，如何提高单线程子系统的响应性，如何确保并发程序执行预期任务，" +
                    "如何提高并发代码的性能和可伸缩性等内容，最后介绍了一些高级主题，如显式锁、原子变量、非阻塞算法以及如何开发自定义的同步工具类。");

            bookMap.put("Hadoop应用开发技术详解", "《Hadoop应用开发技术详解》由资深Hadoop技术专家撰写，系统、全面、" +
                    "深入地讲解了Hadoop开发者需要掌握的技术和知识，包括HDFS的原理和应用、" +
                    "Hadoop文件I/O的原理和应用、MapReduce的原理和高级应用、" +
                    "MapReduce的编程方法和技巧，以及Hive、HBase和Mahout等技术和工具的使用。" +
                    "并且提供大量基于实际生产环境的案例，实战性非常强。");

            bookMap.put("Hadoop核心技术", "这是一本技术深度与企业实践并重的著作，" +
                    "由百度顶尖的Hadoop技术工程师撰写，是百度Hadoop技术实践经验的总结。" +
                    "《Hadoop核心技术》从使用、实现原理、" +
                    "运维和开发4个方面对Hadoop的核心技术进行了深入的讲解");

            bookMap.put("Hadoop YARN权威指南", "《Hadoop YARN权威指南》由YARN的创建和开发团队亲笔撰写，Altiscale的CEO作序鼎力推荐，" +
                    "是使用Hadoop YARN建立分布式、大数据应用的权威指南。书中利用多个实例，详细介绍Hadoop YARN的安装和管理，" +
                    "以帮助用户使用YARN进行应用开发，并在YARN上运行除了MapReduce之外的新框架。\n" +
                    "《Hadoop YARN权威指南》共12章，第1章讲述Apache Hadoop YARN产生和发展的历史；" +
                    "第2章讲解在单台机器（工作站、服务器或笔记本电脑）上快速安装Hadoop 2.0；" +
                    "第3章介绍Apache Hadoop YARN资源管理器；第4章简要介绍YARN组件的功能，帮助读者开始深入了解YARN；" +
                    "第5章详细讲解YARN的安装方法，包括一个基于脚本的手动安装，以及使用Apache Ambari基于GUI的安装；" +
                    "第6章讲述对YARN集群的管理，涉及一些基本的YARN管理场景，介绍如何利用Nagios和Ganglia监控集群，论述对JVM的监视，并介绍Ambari的管理界面；" +
                    "第7章深入探究YARN的架构，向读者展示YARN的内部工作原因；" +
                    "第8章深入讨论Capacity调度器；第9章描述基于现有MapReduce的应用程序如何继续工作以及利用YARN的优势；" +
                    "第10章通过创建一个JBoss Application Server集群的过程，讲述如何构建一个YARN应用程序；" +
                    "第11章描述建立在YARN上的典型示例程序distributed shell的使用和内部情况；" +
                    "第12章总结运行在YARN上的新兴开源框架。最后提供6个附录，" +
                    "包括补充内容和代码下载、YARN的安装脚本、YARN管理脚本、Nagios模块、资源及其他信息、HDFS快速参考");


            bookMap.put("C++面向对象程序设计", "本书采用C++语言来讲解面向对象编程，" +
                    "在介绍C++语法的基础上，还引入了数据结构、设计模式等内容。" +
                    "全书篇章结构精良、组织有序、概念清晰，围绕教学需求展开内容，程序文档形式一致，" +
                    "为学生日后在学术界和专业领域承担程序设计方面的工作打好了基础。");

            bookMap.put("大规模C++程序设计", "《大规模C++程序设计》由世界级软件开发大师John Lakos亲笔撰写，" +
                    "是C++程序设计领域最有影响力的著作之一。作者结合自己多年从事大规模C++项目的开发经验，" +
                    "详细介绍了大规模C++程序设计涉及的一系列概念、理论、原理、设计规则及编程规范，" +
                    "并通过大量真实世界的编程示例，深入解析物理设计与逻辑设计的一些新概念和新理论，" +
                    "阐明在开发大型和超大型C++软件项目时应该遵循的一系列设计规则，" +
                    "论述了设计具有易测试、易维护和可重用等特性的高质量大规模C++软件产品的方法。");

            bookMap.put("大量化创新：研发投入驱动增长的秘密", "研发生产率的传统测算方法只是简单地将研发投入与利润或市场价值挂钩。" +
                    "这让管理者很难了解企业在研发领域的投入是否达到合理标准，更别提让他们的资金投入合理化了。" +
                    "本书提出了一个新的测算方式——RQ（研商），依据经典测算公式，" +
                    "管理者能够做出自己的判断。该测算方法也能够让管理者评估出他们研发投入的理想额度。 " +
                    "RQ是一个非常精确的指标，企业可以依此评估出利润和股价走势。如果企业根据RQ调整研发预算，" +
                    "这将对企业市值产生巨大的积极影响。企业如何知道它可以从研发项目里获得何种回报？搞研发比参与市场竞争更有利吗？" +
                    "企业应该为研发投入多少钱？为了提高投资效率，企业应该怎么做？");

            bookMap.put("量化投资——MATLAB数据挖掘技术与实践", "全书内容分为三篇。第一篇为基础篇，" +
                    "主要介绍量化投资与数据挖掘的关系，以及数据挖掘的概念、实现过程、主要内容、" +
                    "主要工具等内容。第二篇为技术篇，系统介绍了数据挖掘的相关技术及这些技术在量化投资中的应用，" +
                    "主要包括数据的准备、数据的探索、关联规则方法、数据回归方法、分类方法、" +
                    "聚类方法、预测方法、诊断方法、时间序列方法、智能优化方法等内容。第三篇为实践篇，" +
                    "主要介绍数据挖掘技术在量化投资中的综合应用实例，包括统计套利策略的挖掘与优化、" +
                    "配对交易策略的挖掘与实现、数据挖掘在股票程序化交易中的综合应用，以及基于数据挖掘技术的量化交易系统的构建。" +
                    "本书的读者对象为从事投资、数据挖掘、数据分析、数据管理工作的专业人士；" +
                    "金融、经济、管理、统计等专业的教师和学生；希望学习MATLAB的广大科研人员、学者和工程技术人员。");

        }
        // 1，使用IKAnalyzer中文分词器，对于书籍的简介进行分词
        HashMap<String, Multiset<String>> tokenMap = new HashMap<>();
        for (Map.Entry<String, String> entry : bookMap.entrySet()) {
            Multiset<String> words = ConcurrentHashMultiset.create();
            Analyzer analyzer = new IKAnalyzer(true);
            TokenStream ts = analyzer.tokenStream("body", entry.getValue());
            ts.reset();
            while (ts.incrementToken()) {
                String word = ts.getAttribute(CharTermAttribute.class).toString();
                // 1.1 去掉长度为1的字符， 因为 我 的  得 地 这种词对文档聚类不仅不起到作用
                //     反而会影响正确性
                if (word.length() <= 1) {
                    continue;
                }
                words.add(word);
            }
            tokenMap.put(entry.getKey(), words);
        }
        // 2，使用FeatureVectorEncoder编码把中文分词，转换为Vector
        FeatureVectorEncoder encoder = new StaticWordValueEncoder("body");
        // 3，使用NamedVector做外层包裹，这样让输出结果时，方便知道是哪一本书
        List<NamedVector> bookVectors = new ArrayList<>();
        for (Map.Entry<String, Multiset<String>> entry : tokenMap.entrySet()) {
            // 3.1 因为中文分词，分出来的值不是连续的，使用RandomAccessSparseVector最合适
            //     这里设置为1000，只有1000个分词起作用
            Vector bookVector = new RandomAccessSparseVector(1000);
            Multiset<String> words = entry.getValue();
            for (String word : words.elementSet()) {
                // 3.2 使用TF-IDF计算出每个词的特征值
                encoder.addToVector(word, calcTFIDF(word, words, tokenMap), bookVector);
            }
            NamedVector bookNameVector = new NamedVector(bookVector, entry.getKey());
            bookVectors.add(bookNameVector);
        }
        return bookVectors;
    }

    public static void main(String[] args) throws IOException {
        // 1, 加载数据
        List<NamedVector> bookVectors = loadData();
        List<Cluster> clusterModels = new ArrayList<>();
        for (int i = 0; i < 4; ++i) {
            Vector vec = bookVectors.get(i);
            // 这里使用余弦计算距离
            clusterModels.add(new Kluster(vec, i, new CosineDistanceMeasure()));
        }
        // 3，构建聚类算法
        KMeansClusteringPolicy kMeansClusteringPolicy = new KMeansClusteringPolicy();
        ClusterClassifier classifier = new ClusterClassifier(clusterModels, kMeansClusteringPolicy);
        kMeansClusteringPolicy.update(classifier);
        // 4， 运行算法，这里默认迭代10次
        for (int i = 0; i < 100; ++i) {
            for (NamedVector vector : bookVectors) {
                Vector probabilities = classifier.classify(vector);
                Vector weights = kMeansClusteringPolicy.select(probabilities);
                for (Vector.Element e : weights.nonZeroes()) {
                    int index = e.index();
                    classifier.train(index, vector, weights.get(index));
                }
            }
            // 4.1 每次簇中心迭代完毕，计算先选出来簇中心的变化大小
            classifier.close();
            classifier.getPolicy().update(classifier);
            List<Cluster> models = classifier.getModels();
            boolean bBreak = false;
            // 4.2 当簇中心的变化不大时，推出计算
            for (Cluster cluster : models) {
                if (cluster.isConverged()) {
                    bBreak = true;
                    break;
                }
            }
            if (bBreak) {
                break;
            }
        }
        // 5，根据计算出来的簇中心，计算各个点到各个簇中心的距离，然后和离自己最新的簇中心
        //    成为一簇
        for (NamedVector vector : bookVectors) {
            Vector pdfPerCluster = classifier.classify(vector);
            if (pdfPerCluster.maxValue() >= 0) {
                System.out.printf("name = %s clusterId = %d value = %f\n", vector.getName(), pdfPerCluster.maxValueIndex(), pdfPerCluster.maxValue());
            }
        }


    }


    // 优化
//        // 4，目前应知道有Java， Hadoop，C++， 量化 这四个簇，
//        //    为了提高准确性，直接选中以Java，Hadoop，C++，量化其中的一本书籍作为初始簇中心
//        clusterModels.add(new Kluster(bookVectors.get(0), 0, new CosineDistanceMeasure()));
//        clusterModels.add(new Kluster(bookVectors.get(1), 1, new CosineDistanceMeasure()));
//        clusterModels.add(new Kluster(bookVectors.get(3), 2, new CosineDistanceMeasure()));
//        clusterModels.add(new Kluster(bookVectors.get(7), 3, new CosineDistanceMeasure()));


//    // 计算簇间距离
//    List<Cluster> models = classifier.getModels();
//    //
//    DistanceMeasure measure = new CosineDistanceMeasure();
//    double max = 0;
//    double min = Double.MAX_VALUE;
//    double sum = 0;
//    int count = 0;
//        for (int i = 0; i < models.size(); ++i) {
//        for (int j = 0; j < models.size(); ++j) {
//            double d = measure.distance(models.get(i).getCenter(), models.get(j).getCenter());
//            min = Math.min(d, min);
//            max = Math.max(d, max);
//            sum += d;
//            ++count;
//        }
//    }
//        System.out.printf(
//                "max distance : %f, min distance: % f avg distance: %f \n",
//    max ,
//    min,
//            (sum / count - min / (max - min))
//            );

    /**
     * @param word
     * @param dstDoc
     * @param tokenMap
     * @return
     */
    static double calcTFIDF(String word, Multiset<String> dstDoc, HashMap<String, Multiset<String>> tokenMap) {
        double tf = dstDoc.count(word) * 1.0 / dstDoc.size();
        double idf = 0;
        int containDoc = 0;
        for (Map.Entry<String, Multiset<String>> entry : tokenMap.entrySet()) {
            // 3.1 因为中文分词，分出来的值不是连续的，使用RandomAccessSparseVector最合适
            //     这里设置为1000，只有1000个分词起作用

            Multiset<String> words = entry.getValue();
            if (words.contains(word)) {
                ++containDoc;
            }
        }
        idf = Math.log(tokenMap.size() * 1.0 / (containDoc + 1));
        return tf * idf;
    }
}
