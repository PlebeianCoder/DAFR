__all__ = ["PubFigDatasetSize", "PubFigDataset"]

from pathlib import Path
from typing import Union, Optional, List, Tuple, Dict

import torch
from torch.utils.data import DataLoader

from advfaceutil.datasets.faces.base import RESEARCHERS, FaceDatasetSize, FaceDataset
from advfaceutil.utils import split_data


class PubFigDatasetSize(FaceDatasetSize):
    """
    Enumerate the names of individuals for each PubFig dataset size.
    """

    SMALL = [
        "AaronEckhart",
        "BradPitt",
        "DrewBarrymore",
        "CliveOwen",
        "MillaJovovich",
    ]
    LARGE = [
        "DanielRadcliffe",
        "RussellCrowe",
        "TinaFey",
        "AaronEckhart",
        "BradPitt",
        "LindsayLohan",
        "KeiraKnightley",
        "AvrilLavigne",
        "OrlandoBloom",
        "JessicaSimpson",
        "CarsonDaly",
        "JessicaAlba",
        "CharlizeTheron",
        "JohnTravolta",
        "BeyonceKnowles",
        "KateMoss",
        "LeonardoDiCaprio",
        "SalmaHayek",
        "RickyMartin",
        "HughGrant",
        "JodieFoster",
        "NoahWyle",
        "MerylStreep",
        "JasonStatham",
        "LiamNeeson",
        "JeriRyan",
        "NicoleKidman",
        "EvaMendes",
        "OprahWinfrey",
        "RobertDowneyJr",
        "ChristopherWalken",
        "AliciaKeys",
        "AngelaMerkel",
        "DaisyFuentes",
        "JavierBardem",
        "JohnLennon",
        "LivTyler",
        "CindyCrawford",
        "SimonCowell",
        "EdieFalco",
        "TomCruise",
        "MarthaStewart",
        "DrewBarrymore",
        "GeorgeClooney",
        "AntonioBanderas",
        "AngelinaJolie",
        "JoaquinPhoenix",
        "BobDole",
        "CelineDion",
        "SusanSarandon",
        "GaelGarciaBernal",
        "RayRomano",
        "MorganFreeman",
        "SigourneyWeaver",
        "MelGibson",
        "JonStewart",
        "MikhailGorbachev",
        "ElizaDushku",
        "UmaThurman",
        "DenzelWashington",
        "LanceArmstrong",
        "WilliamMacy",
        "CameronDiaz",
        "LucyLiu",
        "VictoriaBeckham",
        "ColinFarrell",
        "HalleBerry",
        "CateBlanchett",
        "KeanuReeves",
        "TyraBanks",
        "AdamSandler",
        "SharonStone",
        "JamesFranco",
        "GwynethPaltrow",
        "JerrySeinfeld",
        "AshtonKutcher",
        "TomHanks",
        "GeneHackman",
        "MonicaBellucci",
        "ReeseWitherspoon",
        "DavidBeckham",
        "AdrianaLima",
        "GeorgeWBush",
        "ReneeZellweger",
        "WillSmith",
        "BenAffleck",
        "JimmyCarter",
        "JohnMalkovich",
        "MichaelDouglas",
        "EliotSpitzer",
        "ColinPowell",
        "KateWinslet",
        "JackNicholson",
        "AlbertoGonzales",
        "SilvioBerlusconi",
        "GillianAnderson",
        "DonaldTrump",
        "ChrisMartin",
        "HarrisonFord",
        "GiseleBundchen",
        "JenniferAniston",
        "ShinzoAbe",
        "KatieCouric",
        "BillClinton",
        "JenniferLopez",
        "MinnieDriver",
        "NicolasSarkozy",
        "JayLeno",
        "MichaelBloomberg",
        "RodStewart",
        "OwenWilson",
        "BruceWillis",
        "JenniferLoveHewitt",
        "MariahCarey",
        "AlecBaldwin",
        "MatthewBroderick",
        "TonyBlair",
        "AnnaKournikova",
        "JeffBridges",
        "BillyCrystal",
        "ClaudiaSchiffer",
        "JamesGandolfini",
        "AshleyJudd",
        "DustinHoffman",
        "ShaniaTwain",
        "RosarioDawson",
        "BrendanFraser",
        "GordonBrown",
        "TigerWoods",
        "StevenSpielberg",
        "NicolasCage",
        "MattDamon",
        "QuincyJones",
        "RalphNader",
        "HollyHunter",
        "CarlaGugino",
        "RosiePerez",
        "PhilipSeymourHoffman",
        "DaveChappelle",
        "NathanLane",
    ]

    FINAL = []


class PubFigDataset(FaceDataset):
    """
    A wrapper on a Multiclass Dataset specifically for PubFig images.
    """

    @staticmethod
    def construct(
        data_directory: Union[str, Path],
        researchers_directory: Union[str, Path],
        size: FaceDatasetSize,
        class_image_limit: int,
        convert_to_bgr: bool = True,
    ) -> "FaceDataset":
        return PubFigDataset(
            data_directory,
            researchers_directory,
            size,
            class_image_limit,
            convert_to_bgr,
        )

    def __init__(
        self,
        data_directory: Union[str, Path],
        researchers_directory: Union[str, Path],
        size: FaceDatasetSize,
        class_image_limit: int,
        convert_to_bgr: bool = True,
        data: Optional[List[Tuple[torch.Tensor, int]]] = None,
        class_index_map: Optional[Dict[str, int]] = None,
    ) -> None:
        """
        Initialise a PubFig dataset with the given data directory and researcher directory
        for the public figures and researchers images respectively.

        :param data_directory: The directory containing the public figure images.
        :param researchers_directory: The directory containing the researcher images.
        :param size: The size of the PubFig dataset to load.
        :param class_image_limit: The maximum number of images per class.
        :param convert_to_bgr: Convert the images to BGR (default is True).
        :param data: The data to load in.
        :param class_index_map: The mapper from class name to index.
        """
        super().__init__(class_image_limit, data, class_index_map, convert_to_bgr)
        self.__data_directory = data_directory
        self.__researchers_directory = researchers_directory
        self.__size = size

        if data is None and class_index_map is None:
            self.load_data(data_directory, size.dataset_names)
            self.load_data(researchers_directory, RESEARCHERS)

    def split_training_testing(
        self,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = 40,
        testing_image_limit: Optional[int] = 5,
    ) -> Tuple["PubFigDataset", "PubFigDataset"]:
        # Store the images per class
        image_classes = {}
        for image, clazz in self._data:
            if clazz in image_classes.keys():
                image_classes[clazz].append((image, clazz))
            else:
                image_classes[clazz] = [(image, clazz)]

        training_data = []
        testing_data = []

        # Split the data for each class
        for clazz, data in image_classes.items():
            training, testing = split_data(
                data, training_ratio, training_image_limit, testing_image_limit
            )
            training_data.extend(training)
            testing_data.extend(testing)

        return PubFigDataset(
            self.__data_directory,
            self.__researchers_directory,
            self.__size,
            training_image_limit,
            self.convert_to_bgr,
            training_data,
            self._class_index_map,
        ), PubFigDataset(
            self.__data_directory,
            self.__researchers_directory,
            self.__size,
            testing_image_limit,
            self.convert_to_bgr,
            testing_data,
            self._class_index_map,
        )

    def split_training_testing_as_loader(
        self,
        batch_size: int,
        training_ratio: Optional[float] = None,
        training_image_limit: Optional[int] = 40,
        testing_image_limit: Optional[int] = 5,
    ) -> Tuple[DataLoader, DataLoader]:
        return super().split_training_testing_as_loader(
            batch_size, training_ratio, training_image_limit, testing_image_limit
        )
