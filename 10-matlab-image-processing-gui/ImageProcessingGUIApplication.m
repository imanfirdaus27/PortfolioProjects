function ImageProcessingGUIApplication()
% =========================================================
% Name: Muhammad Iman Firdaus Bin Md Rostan
% Matric No: M152510008
% Task: Assignment 2
%
% % Information : Load multiple images via file dialog, browse
%            them with Next/Previous buttons, and apply a
%            simple grayscale conversion, enhance, segmentation
%            and morphology.
% =========================================================
%% ---- 1. APPLICATION DATA (shared across callbacks) ----
    images      = {};   % cell array of loaded image data
    originalImages = {};% cell array of original image data
    filenames   = {};   % cell array of file names (for title)
    currentIdx  = 0;    % which image is being shown (0 = none)
%% ---- 2. CREATE THE MAIN FIGURE WINDOW ----------------
    fig = uifigure('Name',     'Image Processing GUI Application', ...
                   'Position', [100 100 820 600], ...
                   'Color',    [0.13 0.13 0.18]);
    % uifigure used to create window
%% ---- 3. TITLE LABEL ----------------------------------
    uilabel(fig, ...
        'Text',             'Image Processing GUI Application', ...
        'Position',         [0 545 820 40], ...
        'FontSize',         22, ...
        'FontWeight',       'bold', ...
        'FontColor',        [0.96 0.76 0.28], ...   % gold
        'HorizontalAlignment', 'center', ...
        'BackgroundColor',  [0.13 0.13 0.18]);
%% ---- 4. AXES (where the image is displayed) ----------
    ax = uiaxes(fig, ...
        'Position',  [180 80 460 440], ...
        'Color',     [0.08 0.08 0.12], ...
        'XColor',    'none', ...
        'YColor',    'none', ...
        'Box',       'off');
    axis(ax, 'image');       % keep aspect ratio
    ax.Toolbar.Visible = 'off';
    
    % Placeholder text inside the axes
    text(ax, 0.5, 0.5, 'No image loaded', ...
        'Units',             'normalized', ...
        'HorizontalAlignment','center', ...
        'FontSize',          14, ...
        'Color',             [0.5 0.5 0.5]);

%% ---- 5. STATUS / FILENAME LABEL ----------------------
    lblStatus = uilabel(fig, ...
        'Text',             'No image loaded', ...
        'Position',         [180 45 460 28], ...
        'FontSize',         11, ...
        'FontColor',        [1 1 1], ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor',  [0.2 0.2 0.25]);

%% ---- 6. COUNTER LABEL (1 of N) -----------------------
    lblCounter = uilabel(fig, ...
        'Text',             '0 / 0', ...
        'Position',         [340 520 140 28], ...
        'FontSize',         11, ...
        'FontColor',        [1 1 1], ...
        'HorizontalAlignment', 'center', ...
        'BackgroundColor',  [0.2 0.2 0.25]);

%% ---- 7. BUTTONS --------------------------------------
    btnStyle = {'FontSize', 13, 'FontWeight', 'bold', ...
                'BackgroundColor', [0.20 0.20 0.28], ...
                'FontColor',       [1 1 1]};

    % --- Load Images ---
    uibutton(fig, 'push', ...
        'Text',     '📂  Load Images', ...
        'Position', [30 520 140 42], ...
        btnStyle{:}, ...
        'BackgroundColor', [0.18 0.52 0.35], ...
        'ButtonPushedFcn', @loadImages);

    % --- Previous ---
    btnPrev = uibutton(fig, 'push', ...
        'Text',     '◀  Previous', ...
        'Position', [30 460 140 42], ...
        btnStyle{:}, ...
        'ButtonPushedFcn', @prevImage);

    % --- Next ---
    btnNext = uibutton(fig, 'push', ...
        'Text',     'Next  ▶', ...
        'Position', [30 405 140 42], ...
        btnStyle{:}, ...
        'ButtonPushedFcn', @nextImage);

    % --- Grayscale ---
    btnGray = uibutton(fig, 'push', ...
        'Text',     '🎨  Grayscale', ...
        'Position', [30 350 140 42], ...
        btnStyle{:}, ...
        'BackgroundColor', [0.35 0.25 0.55], ...
        'ButtonPushedFcn', @convertGrayscale);
    
    % --- Enhance ---
    btnEnhance = uibutton(fig, 'push', ...
    'Text',     '✨ Enhance', ...
    'Position', [30 295 140 42], ...  
    btnStyle{:}, ...
    'BackgroundColor', [0.85 0.55 0.20], ...
    'ButtonPushedFcn', @enhanceImage);
    
    % --- Segmentation ---
    btnSegment = uibutton(fig, 'push', ...
        'Text',     '🔍 Segmentation', ...
        'Position', [30 240 140 42], ...
        btnStyle{:}, ...
        'BackgroundColor', [0.25 0.45 0.65], ...
        'ButtonPushedFcn', @doSegmentation);
    
    % --- Morphology ---
    btnMorph = uibutton(fig, 'push', ...
    'Text',     '🧩 Morphology', ...
    'Position', [30 185 140 42], ...
    btnStyle{:}, ...
    'BackgroundColor', [0.45 0.35 0.25], ...
    'ButtonPushedFcn', @doMorphology);
    
    % --- Reset Color ---
    btnReset = uibutton(fig, 'push', ...
        'Text',     '↺  Reset Color', ...
        'Position', [30 130 140 42], ...
        btnStyle{:}, ...
        'BackgroundColor', [0.50 0.28 0.15], ...
        'ButtonPushedFcn', @resetColor);

    % --- Save Current ---
    uibutton(fig, 'push', ...
        'Text',     '💾  Save Image', ...
        'Position', [30 75 140 42], ...
        btnStyle{:}, ...
        'BackgroundColor', [0.18 0.38 0.60], ...
        'ButtonPushedFcn', @saveImage);

    % --- Clear All ---
    uibutton(fig, 'push', ...
        'Text',     '🗑  Clear All', ...
        'Position', [30 20 140 42], ...
        btnStyle{:}, ...
        'BackgroundColor', [0.55 0.15 0.15], ...
        'ButtonPushedFcn', @clearAll);
% --- Image Info Panel (right side) ---------------------
    pnlInfo = uipanel(fig, ...
        'Title',           ' Image Info ', ...
        'Position',        [655 80 145 440], ...
        'FontSize',        11, ...
        'BackgroundColor', [0.18 0.18 0.25], ...
        'ForegroundColor', [1 1 1]);

    lblInfo = uilabel(pnlInfo, ...
        'Text',             'Load images to see info.', ...
        'Position',         [8 8 128 400], ...
        'FontSize',         10, ...
        'FontColor',        [1 1 1], ...
        'BackgroundColor',  [0.18 0.18 0.25], ...
        'WordWrap',         'on', ...
        'VerticalAlignment','top');
    

%% ---- 8. DISABLE NAV BUTTONS INITIALLY ----------------
    btnPrev.Enable = 'off';
    btnNext.Enable = 'off';
    btnGray.Enable = 'off';
    btnSegment.Enable = 'off';
    btnMorph.Enable = 'off';
    btnReset.Enable = 'off';
    btnEnhance.Enable = 'off';

%% ======================================================
%  CALLBACK FUNCTIONS  (nested so they share workspace)
% ======================================================

% ----------------------------------------------------------
   function loadImages(~, ~)

        [files, path] = uigetfile( ...
            {'*.jpg;*.jpeg;*.png;*.bmp;*.tif;*.tiff;*.gif', ...
             'Image Files (*.jpg,*.png,*.bmp,*.tif,*.gif)'}, ...
            'Select one or more images', ...
            'MultiSelect', 'on');

        if isequal(files, 0)
            return;
        end

        if ischar(files)
            files = {files};
        end

        % Clear old
        images = {};
        originalImages = {};
        filenames = {};
        currentIdx = 0;

        lblStatus.Text = 'Loading...';

        nNew = numel(files);

        for k = 1:nNew
            fullpath = fullfile(path, files{k});

            try
                img = imread(fullpath);

                % FORCE SMALL SIZE
                img = imresize(img, [300 300]);

                images{end+1} = img;
                originalImages{end+1} = img;
                filenames{end+1} = files{k};

            catch
                warning('Failed: %s', files{k});
            end
        end

        currentIdx = 1;

        updateDisplay();
        updateButtons();

        lblStatus.Text = 'Done loading!';
    end
% ----------------------------------------------------------
    function prevImage(~, ~)
        if currentIdx > 1
            currentIdx = currentIdx - 1;
            updateDisplay();
            updateButtons();
        end
    end

    % ----------------------------------------------------------
    function nextImage(~, ~)
        if currentIdx < numel(images)
            currentIdx = currentIdx + 1;
            updateDisplay();
            updateButtons();
        end
    end

    % ----------------------------------------------------------
    function convertGrayscale(~, ~)

        if currentIdx == 0
            return;
        end

        img = images{currentIdx};

        % If already grayscale, skip
        if size(img,3) == 1
            return;
        end

        % Convert once
        grayImg = rgb2gray(img);

        % Replace
        images{currentIdx} = grayImg;

        % FAST display
        imshow(grayImg, 'Parent', ax);
        drawnow limitrate;
    end
    % ----------------------------------------------------------
    function doSegmentation(~, ~)

        if currentIdx == 0
            return;
        end

        img = images{currentIdx};

        % Convert to grayscale only once
        if size(img,3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end

        % FAST threshold
        bw = imbinarize(gray);

        % Display
        imshow(bw, 'Parent', ax);

    end
    % ----------------------------------------------------------
    function doMorphology(~, ~)

        if currentIdx == 0
            return;
        end

        img = images{currentIdx};

        % Convert to grayscale if needed
        if size(img,3) == 3
            gray = rgb2gray(img);
        else
            gray = img;
        end

        % Convert to binary (segmentation step)
        bw = imbinarize(gray);

        % Morphology
        result = imdilate(bw, strel('disk', 5));

        % Display result
        imshow(result, 'Parent', ax);

    end
    % ----------------------------------------------------------
    function enhanceImage(~, ~)

        if currentIdx == 0
            return;
        end

        img = images{currentIdx};

        % Sharpening
        result = imsharpen(img, 'Amount', 5);

        imshow(result, 'Parent', ax);

    end
    % ----------------------------------------------------------
    function resetColor(~, ~)

        if currentIdx == 0
            return;
        end

        % Save current index
        idx = currentIdx;

        % Restore original image
        images{idx} = originalImages{idx};

        % Force to stay on same image
        currentIdx = idx;

        updateDisplay();

    end

    % ----------------------------------------------------------
    function saveImage(~, ~)
    % Saves the currently displayed image to a user-chosen file
        if currentIdx == 0
            uialert(fig, 'No image to save.', 'Error');
            return;
        end
        [saveName, savePath] = uiputfile( ...
            {'*.png','PNG Image'; '*.jpg','JPEG Image'; ...
             '*.bmp','Bitmap'}, ...
            'Save image as ...');
        if isequal(saveName, 0), return; end
        imwrite(images{currentIdx}, fullfile(savePath, saveName));
        uialert(fig, ['Image saved to: ' saveName], 'Saved', ...
                'Icon', 'success');
    end

    % ----------------------------------------------------------
    function clearAll(~, ~)
        answer = uiconfirm(fig, ...
            'Remove ALL loaded images?', 'Confirm Clear', ...
            'Options',     {'Yes, clear', 'Cancel'}, ...
            'DefaultOption', 2, ...
            'CancelOption',  2, ...
            'Icon', 'warning');
        if strcmp(answer, 'Yes, clear')
            images     = {};
            filenames  = {};
            currentIdx = 0;
            cla(ax);
            text(ax, 0.5, 0.5, 'No image loaded', ...
                'Units','normalized','HorizontalAlignment','center', ...
                'FontSize',14,'Color',[0.5 0.5 0.5]);
            lblStatus.Text  = '';
            lblCounter.Text = '';
            lblInfo.Text    = 'Load images to see info.';
            updateButtons();
        end
    end

    % ======================================================
    %  HELPER FUNCTIONS
    % ======================================================

    function updateDisplay()
    % Render the current image and refresh labels
        if currentIdx < 1 || currentIdx > numel(images), return; end
        img = images{currentIdx};
        imshow(img, 'Parent', ax);
        axis(ax, 'image');

        % Status bar
        lblStatus.Text = filenames{currentIdx};

        % Counter
        lblCounter.Text = sprintf('Image %d of %d', ...
                                   currentIdx, numel(images));

        % Info panel
        [h, w, c] = size(img);
        chStr = 'Grayscale';
        if c == 3, chStr = 'RGB (3 channels)'; end
        if c == 4, chStr = 'RGBA (4 channels)'; end
        lblInfo.Text = sprintf( ...
            'File:\n%s\n\nWidth:\n%d px\n\nHeight:\n%d px\n\nType:\n%s\n\nData type:\n%s', ...
            filenames{currentIdx}, w, h, chStr, class(img));
    end

    function updateButtons()

        hasImages = numel(images) > 0;

        % --- ENABLE / DISABLE MAIN BUTTONS ---
        if hasImages
            btnGray.Enable = 'on';
            btnSegment.Enable = 'on';
            btnMorph.Enable = 'on';
            btnReset.Enable = 'on';
            btnEnhance.Enable = 'on';
        else
            btnGray.Enable = 'off';
            btnSegment.Enable = 'off';
            btnMorph.Enable = 'off';
            btnReset.Enable = 'off';
            btnEnhance.Enable = 'off';
        end

        % --- PREVIOUS BUTTON ---
        if hasImages && currentIdx > 1
            btnPrev.Enable = 'on';
        else
            btnPrev.Enable = 'off';
        end

        % --- NEXT BUTTON ---
        if hasImages && currentIdx < numel(images)
            btnNext.Enable = 'on';
        else
            btnNext.Enable = 'off';
        end
    end
end