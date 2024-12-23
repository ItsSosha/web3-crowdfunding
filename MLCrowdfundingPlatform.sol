// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MLCrowdfundingPlatform {

    struct Project {
        uint256 id;
        address creator;
        string title;
        string description;
        uint256 fundingGoal;
        uint256 currentAmount;
        uint256 deadline;
        bool completed;
        uint256 successProbability;
        uint256 backerCount;
        ProjectMetrics metrics;
    }

    struct ProjectMetrics {
        string category;
        uint256 durationDays;
        uint256 descriptionLength;
    }

    struct Contribution {
        address contributor;
        uint256 amount;
        uint256 timestamp;
    }


    event ProjectCreated(
        uint256 indexed id,
        address indexed creator,
        string title,
        uint256 fundingGoal,
        uint256 deadline
    );

    event ContributionMade(
        uint256 indexed projectId,
        address indexed contributor,
        uint256 amount,
        uint256 timestamp
    );

    event ProjectCompleted(
        uint256 indexed projectId,
        bool successful,
        uint256 totalAmount
    );

    event SuccessProbabilityUpdated(
        uint256 indexed projectId,
        uint256 newProbability
    );


    mapping(uint256 => Project) public projects;
    mapping(uint256 => Contribution[]) public projectContributions;
    mapping(address => uint256[]) public creatorProjects;
    mapping(address => uint256[]) public contributorProjects;
    mapping(uint => mapping(address => bool)) private hasContributed;

    uint256 public projectCount;
    address public owner;
    address public mlOracleAddress;

    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this function");
        _;
    }

    modifier onlyOracle() {
        require(msg.sender == mlOracleAddress, "Only ML oracle can call this function");
        _;
    }

    modifier projectExists(uint256 projectId) {
        require(projectId > 0 && projectId <= projectCount, "Project does not exist");
        _;
    }


    constructor(address _mlOracleAddress) {
        owner = msg.sender;
        mlOracleAddress = _mlOracleAddress;
        projectCount = 0;
    }


    function createProject(
        string memory _title,
        string memory _description,
        uint256 _fundingGoal,
        uint256 _durationInDays,
        string memory _category
    ) public returns (uint256) {
        require(_fundingGoal > 0, "Funding goal must be greater than 0");
        require(_durationInDays > 0 && _durationInDays <= 90, "Duration must be between 1 and 90 days");

        projectCount++;

        ProjectMetrics memory metrics = ProjectMetrics(
            _category,
            _durationInDays,
            bytes(_description).length
        );

        projects[projectCount] = Project(
            projectCount,
            msg.sender,
            _title,
            _description,
            _fundingGoal,
            0,
            block.timestamp + (_durationInDays * 1 days),
            false,
            0,
            0,
            metrics
        );

        creatorProjects[msg.sender].push(projectCount);

        emit ProjectCreated(
            projectCount,
            msg.sender,
            _title,
            _fundingGoal,
            block.timestamp + (_durationInDays * 1 days)
        );

        return projectCount;
    }


    function contribute(uint256 _projectId) public payable projectExists(_projectId) {
        Project storage project = projects[_projectId];

        require(!project.completed, "Project is already completed");
        require(block.timestamp < project.deadline, "Project deadline has passed");
        require(msg.value > 0, "Contribution amount must be greater than 0");

        project.currentAmount += msg.value;

        if (!hasContributed[_projectId][msg.sender]) {
            project.backerCount++;
            hasContributed[_projectId][msg.sender] = true;
        }

        projectContributions[_projectId].push(Contribution(
            msg.sender,
            msg.value,
            block.timestamp
        ));

        contributorProjects[msg.sender].push(_projectId);

        emit ContributionMade(_projectId, msg.sender, msg.value, block.timestamp);

        if(project.currentAmount >= project.fundingGoal) {
            completeProject(_projectId, true);
        }
    }


    function updateSuccessProbability(uint256 _projectId, uint256 _probability)
        public
        onlyOracle
        projectExists(_projectId)
    {
        require(_probability <= 100, "Probability must be between 0 and 100");
        projects[_projectId].successProbability = _probability;
        emit SuccessProbabilityUpdated(_projectId, _probability);
    }


    function completeProject(uint256 _projectId, bool _successful) internal {
        Project storage project = projects[_projectId];
        project.completed = true;

        if(_successful) {
            payable(project.creator).transfer(project.currentAmount);
        } else {

            for (uint i = 0; i < projectContributions[_projectId].length; i++) {
                Contribution memory contribution = projectContributions[_projectId][i];
                payable(contribution.contributor).transfer(contribution.amount);
            }
        }

        emit ProjectCompleted(_projectId, _successful, project.currentAmount);
    }


    function getProject(uint256 _projectId) public view projectExists(_projectId)
        returns (
            string memory title,
            string memory description,
            uint256 fundingGoal,
            uint256 currentAmount,
            uint256 deadline,
            bool completed,
            uint256 successProbability,
            uint256 backerCount
        )
    {
        Project memory project = projects[_projectId];
        return (
            project.title,
            project.description,
            project.fundingGoal,
            project.currentAmount,
            project.deadline,
            project.completed,
            project.successProbability,
            project.backerCount
        );
    }

    function getProjectMetrics(uint256 _projectId) public view projectExists(_projectId)
        returns (ProjectMetrics memory)
    {
        return projects[_projectId].metrics;
    }

    function getProjectContributions(uint256 _projectId) public view projectExists(_projectId)
        returns (Contribution[] memory)
    {
        return projectContributions[_projectId];
    }

    function getUserProjects(address _user) public view
        returns (uint256[] memory created, uint256[] memory contributed)
    {
        return (creatorProjects[_user], contributorProjects[_user]);
    }
}
