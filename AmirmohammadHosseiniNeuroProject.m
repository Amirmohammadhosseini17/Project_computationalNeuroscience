%AmirmohammadHosseini 402617899 Computational Neuroscience's project14022 IUST

% تعریف تعداد نورون‌ها
num_neurons = length(pepANA.listOfResults);

% آرایه‌ای برای ذخیره Fano Factor هر نورون
fano_factors_neurons = zeros(num_neurons, 1);
cv_neurons = zeros(num_neurons, 1); % ذخیره Coefficient of Variation برای هر نورون
silhouette_scores_kmeans = zeros(num_neurons, 1); % ذخیره Silhouette Score برای k-means
silhouette_scores_pca = zeros(num_neurons, 1); % ذخیره Silhouette Score برای PCA

for neuron_idx = 1:num_neurons
    % استخراج داده‌های اسپایک برای هر نورون
    q = pepANA.listOfResults{neuron_idx}.repeat{1}.data{13}{2};
    q = double(q); % تبدیل داده‌ها به نوع double
    
    % بررسی داده‌های q
    if isempty(q)
        error('q is empty.');
    end
    
    % اعمال محاسبات اولیه بر روی q
    figure;
    plot(q, '');
    title(['Data for Neuron ' num2str(neuron_idx)]);
    xlabel('Time');
    ylabel('Data');
    
    [u, s, v] = svd(q);
    figure;
    plot(v(:,1), v(:,2), '.', 'markersize', 10); hold on; plot(0, 0, 'r.', 'markersize', 25);
    title(['Principal Components for Neuron ' num2str(neuron_idx)]);
    xlabel('PC1');
    ylabel('PC2');
    
    idx_kmeans = kmeans(q', 2); % خوشه‌بندی موج‌ها در 2 دسته با استفاده از k-means
    idx_pca = kmeans(v(:,1:2), 2); % خوشه‌بندی بر اساس PCA
    
    idx1_kmeans = find(idx_kmeans == 1);
    idx2_kmeans = find(idx_kmeans == 2);
    
    idx1_pca = find(idx_pca == 1);
    idx2_pca = find(idx_pca == 2);
    
    % بررسی اینکه idx1 و idx2 تهی نباشند
    if isempty(idx1_kmeans) || isempty(idx2_kmeans)
        error('One of the k-means clusters is empty.');
    end
    
    if isempty(idx1_pca) || isempty(idx2_pca)
        error('One of the PCA clusters is empty.');
    end
    
    % نمایش خوشه‌ها با k-means
    figure;
    subplot(1, 2, 1)
    plot(v(idx1_kmeans, 1), v(idx1_kmeans, 2), 'b.', 'markersize', 10); hold on; plot(0, 0, 'r.', 'markersize', 25);
    plot(v(idx2_kmeans, 1), v(idx2_kmeans, 2), 'g.', 'markersize', 10);
    title(['k-means Clusters for Neuron ' num2str(neuron_idx)]);
    xlabel('PC1');
    ylabel('PC2');
    legend('Cluster 1', 'Cluster 2');
    
    subplot(1, 2, 2)
    errorbar(mean(q(:, idx2_kmeans)'), std(q(:, idx2_kmeans)'), 'g'); 
    hold on; % رسم میانگین موج‌ها به همراه ±1 انحراف معیار
    errorbar(mean(q(:, idx1_kmeans)'), std(q(:, idx1_kmeans)'), 'b');
    title(['Mean Waveforms for k-means Clusters for Neuron ' num2str(neuron_idx)]);
    xlabel('Time');
    ylabel('Mean +/- SD');
    legend('Cluster 2', 'Cluster 1');
    xlim([0 49]); % تنظیم محدوده محور x
    
    % نمایش خوشه‌ها با PCA
    figure;
    subplot(1, 2, 1)
    plot(v(idx1_pca, 1), v(idx1_pca, 2), 'b.', 'markersize', 10); hold on; plot(0, 0, 'r.', 'markersize', 25);
    plot(v(idx2_pca, 1), v(idx2_pca, 2), 'g.', 'markersize', 10);
    title(['PCA Clusters for Neuron ' num2str(neuron_idx)]);
    xlabel('PC1');
    ylabel('PC2');
    legend('Cluster 1', 'Cluster 2');
    
    subplot(1, 2, 2)
    errorbar(mean(q(:, idx2_pca)'), std(q(:, idx2_pca)'), 'g'); 
    hold on; % رسم میانگین موج‌ها به همراه ±1 انحراف معیار
    errorbar(mean(q(:, idx1_pca)'), std(q(:, idx1_pca)'), 'b');
    title(['Mean Waveforms for PCA Clusters for Neuron ' num2str(neuron_idx)]);
    xlabel('Time');
    ylabel('Mean +/- SD');
    legend('Cluster 2', 'Cluster 1');
    xlim([0 49]); % تنظیم محدوده محور x
    
    % محاسبه Fano Factor برای این نورون
    T = 3/90; % مدت زمان موثر هر فریم به ثانیه
    spk = pepANA.listOfResults{neuron_idx}.repeat{1}.data{13}{1};
    spk = spk - 60e-3; % محاسبه فریم‌هایی که 60 میلی‌ثانیه قبل از اسپایک روی صفحه بودند
    frame_idx = floor(spk / T);
    
    % رسم قطار اسپایک
    figure;
    hold on;
    for i = 1:length(spk)
        line([spk(i) spk(i)], [0 1], 'Color', 'y'); % رسم هر اسپایک به صورت یک خط عمودی
    end
    xlabel('Time (s)');
    ylabel('Spike');
    title(['Spike Train for Neuron ' num2str(neuron_idx)]);
    hold off;
    
    % محاسبه Interspike Interval
    isi = diff(spk); % محاسبه Interspike Interval
    
    % نمایش histogram برای ISI
    figure;
    histogram(isi);
    xlabel('Interspike Interval (s)');
    ylabel('Frequency');
    title(['Histogram of Interspike Intervals for Neuron ' num2str(neuron_idx)]);
    
    % منحنی tuning curve
    stimuli = unique(frame_idx); % فرض بر این که frame_idx شامل اندیس‌های محرک است
    spike_counts = zeros(size(stimuli));
    
    for i = 1:length(stimuli)
        spike_counts(i) = sum(frame_idx == stimuli(i)); % تعداد اسپایک‌ها برای هر محرک
    end
    
    mean_spike_rate = spike_counts / length(spk); % نرخ اسپایک برای هر محرک
    
    figure;
    plot(stimuli, mean_spike_rate, '-');
    xlabel('Stimulus');
    ylabel('Mean Spike Rate');
    title(['Tuning Curve for Neuron ' num2str(neuron_idx)]);
    
    % محاسبه Fano Factor
    mean_spikes = mean(spike_counts);
    var_spikes = var(spike_counts);
    fano_factor = var_spikes / mean_spikes;
    fprintf('Neuron %d Fano Factor: %.4f\n', neuron_idx, fano_factor);
    fano_factors_neurons(neuron_idx) = fano_factor;
    
    % محاسبه Coefficient of Variation
    cv = std(isi) / mean(isi);
    fprintf('Neuron %d Coefficient of Variation: %.4f\n', neuron_idx, cv);
    cv_neurons(neuron_idx) = cv;
    
    % محاسبه Silhouette Score برای k-means و PCA
    silhouette_scores_kmeans(neuron_idx) = mean(silhouette(q', idx_kmeans));
    silhouette_scores_pca(neuron_idx) = mean(silhouette(v(:,1:2), idx_pca));
    
   end

% رسم هیستوگرام Fano Factor بین نورون‌ها
figure;
histogram(fano_factors_neurons);
xlabel('Fano Factor');
ylabel('Frequency');
title('Histogram of Fano Factors between Neurons');

% رسم هیستوگرام Coefficient of Variation بین نورون‌ها
figure;
histogram(cv_neurons);
xlabel('Coefficient of Variation');
ylabel('Frequency');
title('Histogram of Coefficients of Variation between Neurons');

% رسم هیستوگرام Silhouette Score برای k-means
figure;
histogram(silhouette_scores_kmeans);
xlabel('Silhouette Score (k-means)');
ylabel('Frequency');
title('Histogram of Silhouette Scores for k-means Clustering');

% رسم هیستوگرام Silhouette Score برای PCA
figure;
histogram(silhouette_scores_pca);
xlabel('Silhouette Score (PCA)');
ylabel('Frequency');
title('Histogram of Silhouette Scores for PCA Clustering');


