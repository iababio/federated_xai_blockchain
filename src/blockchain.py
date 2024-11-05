from web3 import Web3

ABI = '''[
    {
        "inputs":[{"internalType":"address","name":"account","type":"address"},
        {"internalType":"address","name":"minter_","type":"address"}], 
        "name":"mint", "outputs":[], "stateMutability":"nonpayable",
        "type":"function"
        }
    ]'''

# Use Web3 to interact with the contract
W3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))
CONTRACT_ADDRESS = "0x6C8f2A135f6ed072DE4503Bd7C4999a1a17F824B"
CONTRACT_ABI = ABI  # ABI loaded from the file

dataset_contract = W3.eth.contract(address=CONTRACT_ADDRESS, abi=CONTRACT_ABI)
