from requests.auth import _basic_auth_str
from requests.auth import _basic_auth_str
import suds
from suds.client import Client
import base64

# clients for everything else:
def prepare_clients(username:str, password:str)->tuple[Client]:
    base64string = base64.b64encode(("{0}:{1}".format(username, password).encode('utf-8'))).decode()
    authentication_header = {
                "WWW-Authenticate": "https://matrikkel.no",
                "Authorization": "Basic %s" % base64string
            }
    #you need your own transport line for each client
    transport = suds.transport.https.HttpAuthenticated(
                username=username,
                password=password
            )
    transport2 = suds.transport.https.HttpAuthenticated(
                username=username,
                password=password
            )
    transportKode = suds.transport.https.HttpAuthenticated(
                username=username,
                password=password
            )
    transportKommune = suds.transport.https.HttpAuthenticated(
                username=username,
                password=password
            )
    transportMatrikkelUnit = suds.transport.https.HttpAuthenticated(
                username=username,
                password=password
            )

    #one client per webpage on matrikkel documentation
    client_buildings = Client("https://matrikkel.no/matrikkelapi/wsapi/v1/BygningServiceWS?WSDL", transport=transport)
    client_buildings.set_options(headers=authentication_header)

    client_objects = Client("https://matrikkel.no/matrikkelapi/wsapi/v1/StoreServiceWS?WSDL", transport=transport2)
    client_objects.set_options(headers=authentication_header)

    clientKodelister = Client("http://matrikkel.no/matrikkelapi/wsapi/v1/KodelisteServiceWS?WSDL", transport=transportKode)
    clientKodelister.set_options(headers=authentication_header)

    clientKommune = Client("https://matrikkel.no/matrikkelapi/wsapi/v1/KommuneServiceWS?WSDL", transport=transportKommune)
    clientKommune.set_options(headers=authentication_header)

    client_matrikkel_unit = Client("http://matrikkel.no/matrikkelapi/wsapi/v1/MatrikkelenhetServiceWS?WSDL", transport = transportMatrikkelUnit)
    client_matrikkel_unit.set_options(headers=authentication_header)

    return [client_buildings, client_objects, clientKodelister, clientKommune, client_matrikkel_unit]
