classdef mySysObj < matlab.System
    % Untitled Add summary here
    %
    % NOTE: When renaming the class name Untitled, the file name
    % and constructor name must be updated to use the class name.
    %
    % This template includes most, but not all, possible properties, attributes,
    % and methods that you can implement for a System object in Simulink.

    % Public, tunable properties

    % Public, non-tunable properties
    properties(Nontunable)
        ChunkSize = 20;
    end


    % Pre-computed constants
    properties(Access = private)
        IncrNbMdl % Naive Bayes incremental classifier
    end

    methods
        % Constructor
        function obj = mySysObj(varargin)
            % Support name-value pair arguments when constructing object
            setProperties(obj,nargin,varargin{:})
        end
    end

    methods(Access = protected)
        %% Common functions
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            initNbMdl = evalin('base', 'initNbMdl');
            obj.IncrNbMdl = incrementalLearner(initNbMdl, MetricsWarmupPeriod = 2000, ...
                MetricsWindowSize = 500, Metrics = "classiferror");
        end

        function [yHat, eRate] = stepImpl(obj,X, y)
            % Implement algorithm. 
            % Calculate y as a function of input u and discrete states.
            yHat = obj.IncrNbMdl.predict(X);
            obj.IncrNbMdl = updateMetricsAndFit(obj.IncrNbMdl, X, y);
            eRate = obj.IncrNbMdl.Metrics{"ClassificationError",:};
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
            initNbMdl = evalin('base', 'initNbMdl');
            obj.IncrNbMdl = incrementalLearner(initNbMdl, MetricsWarmupPeriod = 2000, ...
                MetricsWindowSize = 500, Metrics = "classiferror");
        end

        %% Backup/restore functions
        function s = saveObjectImpl(obj)
            % Set properties in structure s to values in object obj

            % Set public properties and states
            s = saveObjectImpl@matlab.System(obj);

            % Set private and protected properties
            %s.myproperty = obj.myproperty;
        end

        function loadObjectImpl(obj,s,wasLocked)
            % Set properties in object obj to values in structure s

            % Set private and protected properties
            % obj.myproperty = s.myproperty; 

            % Set public properties and states
            loadObjectImpl@matlab.System(obj,s,wasLocked);
        end

        %% Simulink functions
        function ds = getDiscreteStateImpl(obj)
            % Return structure of properties with DiscreteState attribute
            ds = struct([]);
        end

        function flag = isInputSizeMutableImpl(obj,index)
            % Return false if input size cannot change
            % between calls to the System object
            flag = false;
        end

        function [out1, out2] = getOutputSizeImpl(obj)
            % Return size for each output port
            out1 = [obj.ChunkSize, 1];
            out2 = [1, 2];

            % Example: inherit size from first input port
            % out = propagatedInputSize(obj,1);
        end

        function [out1,out2] = getOutputDataTypeImpl(obj)
            % Return data type for each output port
            out1 = "double";
            out2 = "double";

            % Example: inherit data type from first input port
            % out = propagatedInputDataType(obj,1);
        end

        function [out1,out2] = isOutputComplexImpl(obj)
            % Return true for each output port with complex data
            out1 = false;
            out2 = false;

            % Example: inherit complexity from first input port
            % out = propagatedInputComplexity(obj,1);
        end

        function [out1,out2] = isOutputFixedSizeImpl(obj)
            % Return true for each output port with fixed size
            out1 = true;
            out2 = true;

            % Example: inherit fixed-size status from first input port
            % out = propagatedInputFixedSize(obj,1);
        end

        function icon = getIconImpl(obj) %#ok<MANU>
            % Define icon for System block
            icon = ["Incremental","Classifier","(Naive Bayes)"];
            % icon = "My System"; % Example: text icon
            % icon = ["My","System"]; % Example: multi-line text icon
            % icon = matlab.system.display.Icon("myicon.jpg"); % Example: image file icon
        end
    end

    methods(Static, Access = protected)
        %% Simulink customization functions
        function header = getHeaderImpl
            % Define header panel for System block dialog
            header = matlab.system.display.Header(mfilename("class"), ...
                'Title','Incremental Learner',...
                'Text', 'Incremental Naive Bayesian Classification');
        end

        function group = getPropertyGroupsImpl
            % Define property section(s) for System block dialog
            group = matlab.system.display.Section(mfilename("class"));
        end
    end
end
