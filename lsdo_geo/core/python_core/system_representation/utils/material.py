import numpy as np
import xml.etree.ElementTree as ET
import sys

class Material():
    def __init__(self, name = None, density = None, compliance = None, thickness = None):
        self.name = name
        self.density = density
        self.compliance = compliance
        self.thickness = thickness
    
    # https://docs.python.org/3/library/xml.etree.elementtree.html
    def import_xml(self, fname):
        tree = ET.parse(fname)
        root = tree.getroot()

        self.name = root.attrib['name']

        if root.find('density') is not None: 
            self.density = float(root.find('density').text)
            
        if root.find('compliance') is not None:
            self.compliance = np.array(
                [[float(x) for x in i.text.split()] for i in root.find('compliance')]
                )
        
        if root.find('thickness') is not None: 
            self.thickness = float(root.find('thickness').text)

    def export_xml(self, fname):
        root = ET.Element('material')
        root.set('name', self.name)

        if self.density:
            ET.SubElement(root, 'density').text = str(self.density)

        if self.compliance is not None:
            compliance_el = ET.SubElement(root, 'compliance')
            for row in self.compliance:
                ET.SubElement(compliance_el, 'row').text = ' '.join(map(str, row))

        if self.thickness:
            ET.SubElement(root, 'thickness').text = str(self.thickness)

        tree = ET.ElementTree(root)
        if sys.version_info[1] >= 9:
            ET.indent(tree) # makes it pretty
        tree.write(fname)

class IsotropicMaterial(Material):
    def compute_compliance(self, E = None, nu = None, G = None):
            if not None in [E, nu]:
                pass
            elif not None in [G, nu]:
                E = G*2*(1+nu)
            elif not None in [E, G]:
                nu = E/(2*G)-1
            else:
                print('Material properties are uderdefined')

            self.compliance = 1/E*np.array(
                [[1, -nu, -nu, 0, 0, 0],
                [-nu, 1, -nu, 0, 0, 0],
                [-nu, -nu, 1, 0, 0, 0],
                [0, 0, 0, 1+nu, 0, 0],
                [0, 0, 0, 0, 1+nu, 0],
                [0, 0, 0, 0, 0, 1+nu]]
            )

    def from_compliance(self):
        E = 1/self.compliance[0,0]
        nu = -self.compliance[0,1]*E
        return E, nu
    
class TransverseMaterial(Material):
    def compute_compliance(self, EA, ET, vA, vT, GA):
            # E1 = EA
            # E2 = E3 = ET
            # v12 = v13 = vA
            # v23 = vT
            # G12 = G13 = GA

            GT = ET/(2*(1+vT)) # = G23

            self.compliance = np.array(
                [[1/ET, -vT/ET, -vA/EA, 0, 0, 0],
                [-vT/ET, 1/ET, -vA/EA, 0, 0, 0],
                [-vA/EA, -vA/EA, 1/EA, 0, 0, 0],
                [0, 0, 0, 1/GA, 0, 0],
                [0, 0, 0, 0, 1/GA, 0],
                [0, 0, 0, 0, 0, 1/GT]]
            )

    def from_compliance(self):
        ET = self.compliance[0,0]
        EA = self.compliance[5,5]
        
        return EA, ET#, nuA, nuT, GA

